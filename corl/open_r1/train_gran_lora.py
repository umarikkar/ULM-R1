"""Ground the gran-LoRA "granularity knob" (Stage-0, supervised).

Trains ONE LoRA on the Janus captioner (i2t path) so that the LoRA **scale alpha**
controls caption granularity, under a single FIXED neutral prompt. alpha=0 (LoRA
off) = the base model's default caption; higher/lower alpha => the granularity we
mapped to it. We supervise, per image, each alpha toward its cached level caption
(scale-consistency), so the alpha-axis is grounded rather than hoped-for.

    fixed neutral prompt  ─►  Janus + gran-LoRA·alpha  ─►  caption  (target = l_k)

Data: PubMedVision_CachedCaptions_Levels.json (cached_captions_l1/l2/l3[/l1_meta]).
Loss: LM cross-entropy on the caption tokens only (prompt + image masked).

Launch:  bash corl/scripts/train_gran_lora.sh
Validate afterwards by sweeping alpha and checking caption length/detail is
monotone (that is the real grounding check).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

from janus.models import VLChatProcessor
from corl.open_r1.janus_tokenizer_fix import load_fast_tokenizer

NEUTRAL_PROMPT = "Describe this medical image."


class LevelsDataset(Dataset):
    def __init__(self, rows, data_dir, levels):
        self.rows = rows
        self.data_dir = data_dir
        self.levels = levels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img_rel = r["image"][0] if isinstance(r["image"], (list, tuple)) else r["image"]
        return {
            "path": os.path.join(self.data_dir, img_rel),
            "caps": {lv: r[f"cached_captions_{lv}"] for lv in self.levels},
        }


def collate(batch):
    return batch  # list of dicts; we build tensors per-alpha in the loop


def build_level_inputs(processor, imgs, caps_for_level, device):
    """Build left-padded (input_embeds inputs, labels) for one granularity level.

    imgs: list[PIL], caps_for_level: list[str] aligned with imgs.
    Labels supervise only the answer (caption) tokens; prompt+image are -100.
    """
    tok = processor.tokenizer
    prepares, answer_lens = [], []
    for img, cap in zip(imgs, caps_for_level):
        user = f"<image_placeholder>\n{NEUTRAL_PROMPT}"
        conv_full = [{"role": "<|User|>", "content": user},
                     {"role": "<|Assistant|>", "content": cap}]
        conv_prompt = [{"role": "<|User|>", "content": user},
                       {"role": "<|Assistant|>", "content": ""}]
        full_str = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conv_full, sft_format=processor.sft_format, system_prompt="")
        prompt_str = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conv_prompt, sft_format=processor.sft_format, system_prompt="")
        alen = len(tok.encode(full_str)) - len(tok.encode(prompt_str))
        prepares.append(processor.process_one(conversations=conv_full, images=[img]))
        answer_lens.append(max(int(alen), 1))

    batched = processor.batchify(prepares).to(device)
    input_ids = batched.input_ids                      # [B, T], LEFT-padded
    T = input_ids.shape[1]
    labels = torch.full_like(input_ids, -100)
    for i, alen in enumerate(answer_lens):
        labels[i, T - alen:] = input_ids[i, T - alen:]
    # position_ids for left padding (else Llama mis-positions padded seqs)
    am = batched.attention_mask.long()
    position_ids = am.cumsum(-1) - 1
    position_ids.masked_fill_(am == 0, 0)
    return batched, labels, position_ids


def set_gran_scale(model, base_scaling, alpha):
    """Set every gran-LoRA layer's scale to alpha * base (alpha=0 -> LoRA off)."""
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            for adp in m.scaling:
                m.scaling[adp] = base_scaling[(name, adp)] * alpha


class GranCaptioner(torch.nn.Module):
    """Thin wrapper so DDP.forward runs the whole i2t forward (embeds + LM + loss)
    and gradient sync is set up correctly (calling language_model directly would
    bypass DDP's prepare_for_backward)."""

    def __init__(self, janus):
        super().__init__()
        self.janus = janus

    def forward(self, input_ids, pixel_values, images_seq_mask, images_emb_mask,
                attention_mask, position_ids, labels):
        emb = self.janus.prepare_inputs_embeds(
            input_ids=input_ids, pixel_values=pixel_values,
            images_seq_mask=images_seq_mask, images_emb_mask=images_emb_mask)
        out = self.janus.language_model(
            inputs_embeds=emb, attention_mask=attention_mask,
            position_ids=position_ids, labels=labels)
        return out.loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/Janus-Pro-1B")
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--levels", default="l1,l2,l3")
    # alpha mapping: default base is detailed, so higher alpha => coarser.
    ap.add_argument("--alpha_map", default="l1:1.0,l2:0.6,l3:0.3",
                    help="level:alpha pairs; alpha scales the gran-LoRA (0=off=base).")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--exclude_ids_json", default="")
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--log_steps", type=int, default=10)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    args = ap.parse_args()

    levels = args.levels.split(",")
    amap = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in args.alpha_map.split(",")}
    for lv in levels:
        assert lv in amap, f"alpha_map missing level {lv}"

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    is_main = rank == 0

    # ---- data ----
    with open(args.data_json) as f:
        rows = json.load(f)
    if args.exclude_ids_json:
        with open(args.exclude_ids_json) as f:
            raw = json.load(f)
        excl = {r["id"] if isinstance(r, dict) else r for r in raw}
        rows = [r for r in rows if r.get("id") not in excl]
    if args.max_samples:
        rows = rows[: args.max_samples]
    if is_main:
        print(f"[data] {len(rows)} rows | levels={levels} | alpha_map={amap}", flush=True)

    # ---- model + processor ----
    processor = VLChatProcessor.from_pretrained(args.model)
    processor.system_prompt = ""
    # Janus' declared slow LlamaTokenizer mangles encode (drops spaces on this
    # byte-level BPE vocab); swap in a correctly-encoding fast tokenizer so the
    # supervised caption tokens keep their spaces. See janus_tokenizer_fix.
    processor.tokenizer = load_fast_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)

    # gran-LoRA on the LLM (captioner path only).
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type=None,
    )
    # Wrap only the language model so vision/gen towers stay clean & frozen.
    model.language_model = get_peft_model(model.language_model, peft_cfg)
    model = model.to(device).train()
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)   # only gran-LoRA trains
    if args.gradient_checkpointing:
        model.language_model.gradient_checkpointing_enable()
        model.language_model.enable_input_require_grads()

    # capture each LoRA layer's base scaling so alpha=1 reproduces the config scale
    base_scaling = {}
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            for adp in m.scaling:
                base_scaling[(name, adp)] = float(m.scaling[adp])

    trainable = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        print(f"[model] trainable params: {sum(p.numel() for p in trainable):,}", flush=True)

    captioner = GranCaptioner(model)
    fwd = captioner
    if world_size > 1:
        fwd = torch.nn.parallel.DistributedDataParallel(
            captioner, device_ids=[local_rank], find_unused_parameters=True)

    # ---- loader ----
    ds = LevelsDataset(rows, args.data_dir, levels)
    sampler = DistributedSampler(ds, shuffle=True) if world_size > 1 else None
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        shuffle=(sampler is None), num_workers=4,
                        collate_fn=collate, drop_last=True)

    steps_per_epoch = len(loader)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    warmup = int(args.warmup_ratio * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, warmup)) *
        (0.5 * (1 + torch.cos(torch.tensor(
            3.14159 * max(0, s - warmup) / max(1, total_steps - warmup))).item())))

    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            imgs = [Image.open(b["path"]).convert("RGB") for b in batch]
            opt.zero_grad(set_to_none=True)
            loss_log = {}
            # scale-consistency: same images at every alpha, one backward each.
            for li, lv in enumerate(levels):
                set_gran_scale(model, base_scaling, amap[lv])
                caps = [b["caps"][lv] for b in batch]
                batched, labels, position_ids = build_level_inputs(
                    processor, imgs, caps, device)
                loss = fwd(input_ids=batched.input_ids,
                           pixel_values=batched.pixel_values,
                           images_seq_mask=batched.images_seq_mask,
                           images_emb_mask=batched.images_emb_mask,
                           attention_mask=batched.attention_mask,
                           position_ids=position_ids, labels=labels)
                (loss / len(levels)).backward()
                loss_log[lv] = float(loss.detach())

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); step += 1

            if is_main and step % args.log_steps == 0:
                dt = time.perf_counter() - t0
                ls = " ".join(f"{k}={v:.3f}" for k, v in loss_log.items())
                print(f"[step {step}/{total_steps}] {ls} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"({step / max(dt,1e-3):.2f} it/s)", flush=True)
            if is_main and step % args.save_steps == 0:
                set_gran_scale(model, base_scaling, 1.0)  # save at canonical scale
                model.language_model.save_pretrained(out_dir)
                print(f"[save] adapter -> {out_dir} (step {step})", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    if is_main:
        set_gran_scale(model, base_scaling, 1.0)
        model.language_model.save_pretrained(out_dir)
        print(f"[done] saved gran-LoRA -> {out_dir}", flush=True)
    if world_size > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
