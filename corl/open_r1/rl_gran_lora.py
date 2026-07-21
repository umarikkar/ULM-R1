"""RL-refine the gran-LoRA granularity knob with GRPO on a length-target reward.

Stage-1 (see HANDOFF_GRANULARITY_RL.md, revised). A FRESH rank-8 LoRA on the Janus
captioner is trained so its scale alpha in [0,1] is a granularity knob: alpha=1 ->
concise, alpha=0 -> verbose. We do NOT supervise caption text. Instead:

  per step, per image:
    1. sample alpha ~ U(0,1)
    2. set gran-LoRA scale = alpha, sample K captions (do_sample, temperature)
    3. reward each = -|log len_tok - log_target(alpha)|  - w_rep * rep_frac   (gran_reward)
    4. GRPO advantage within the K-group: A_i = r_i - mean(r)   (mean-only baseline)
    5. policy grad on token logprobs at scale alpha, + beta * KL(policy || base)
       where "base" logprobs come from the SAME model at scale 0 (LoRA off) --
       no separate reference model needed. KL uses the k3 estimator (trl-style).

Only lora_ params on model.language_model train; everything else frozen.
Faithfulness is held by the KL-to-base anchor, not the reward.

Launch:  bash corl/scripts/rl_gran_lora.sh
Validate: corl/eval/sweep_gran_alpha.py on the saved adapter (alpha should move
length smoothly; check captions stay faithful vs the base).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
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
from corl.open_r1 import gran_reward as R

# Fixed anchor prompt (system + user), shared across ALL alpha and BYTE-IDENTICAL in
# the rollout and the scale-0 KL reference (both go through _prompt_embeds), so it
# anchors the whole run. Design (see corl/eval/select_anchor_prompt.py sweep):
#  * SYSTEM = Janus' trained default identity + a light "medical imaging" edge. Kept
#    task-agnostic so the same system prompt can serve other tasks (the captioning
#    constraint does NOT live here).
#  * USER = the task instruction + an anti-hallucination constraint. The "describe ALL
#    findings" framing was DROPPED — on the 1B base it pushes enumeration -> fabrication.
ANCHOR_SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant with expertise in medical "
    "imaging. You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)
USER_PROMPT = (
    "Describe this medical image. Describe only what is visibly present; do not "
    "speculate or state findings that are not directly supported by what you see."
)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class ImageDataset(Dataset):
    """Just images + ids; RL never looks at the cached captions (no text targets)."""

    def __init__(self, rows, data_dir):
        self.rows = rows
        self.data_dir = data_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img_rel = r["image"][0] if isinstance(r["image"], (list, tuple)) else r["image"]
        return {"id": r.get("id"), "path": os.path.join(self.data_dir, img_rel)}


def collate(batch):
    return batch


def set_gran_scale(model, base_scaling, alpha):
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            for adp in m.scaling:
                m.scaling[adp] = base_scaling[(name, adp)] * alpha


# --------------------------------------------------------------------------- #
# rollout + logprob helpers  (single image -> a group of K completions)
# --------------------------------------------------------------------------- #
def _prompt_embeds(janus, processor, img, device):
    """image + anchor-prompt embeds for ONE image -> (embeds[1,P,H], attn[1,P]).

    Uses processor.system_prompt (set to the fixed anchor in main), so rollout and
    the scale-0 KL reference share a byte-identical prompt."""
    conv = [{"role": "<|User|>", "content": f"<image_placeholder>\n{USER_PROMPT}"},
            {"role": "<|Assistant|>", "content": ""}]
    # single-conversation mode: conversations[0] is a dict (not a list), so the
    # processor takes the else-branch and `images` must be a FLAT list of PIL images.
    prep = processor(conversations=conv, images=[img], force_batchify=True).to(device)
    emb = janus.prepare_inputs_embeds(
        input_ids=prep.input_ids, pixel_values=prep.pixel_values,
        images_seq_mask=prep.images_seq_mask, images_emb_mask=prep.images_emb_mask)
    return emb, prep.attention_mask


@torch.inference_mode()
def sample_group(janus, processor, img, device, *, k, max_new_tokens, temperature, top_p, eos):
    """Sample K captions for one image at the CURRENTLY-set gran scale.

    Returns list of completion token-id LongTensors [len_i] (eos trimmed)."""
    emb, attn = _prompt_embeds(janus, processor, img, device)
    out = janus.language_model.generate(
        inputs_embeds=emb.expand(k, -1, -1),
        attention_mask=attn.expand(k, -1),
        max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        pad_token_id=eos, bos_token_id=processor.tokenizer.bos_token_id, eos_token_id=eos)
    # with inputs_embeds, generate() returns ONLY the newly generated tokens.
    comps = []
    for row in out:
        ids = row.tolist()
        if eos in ids:
            ids = ids[: ids.index(eos)]      # trim at first eos
        comps.append(torch.tensor(ids, dtype=torch.long, device=device))
    return comps


def token_logprobs(janus, processor, img, comp_ids, device, *, requires_grad):
    """Per-token logprob of a completion under the CURRENTLY-set gran scale.

    Teacher-forces [image+prompt embeds] ++ [completion token embeds] and reads the
    logprob of each completion token. Returns tensor [len] (grad iff requires_grad)."""
    # comp_ids came from generate() under inference_mode -> it is an "inference tensor"
    # that cannot be saved for backward; clone it to a normal tensor for the grad path.
    comp_ids = comp_ids.clone()
    ctx = torch.enable_grad() if requires_grad else torch.inference_mode()
    with ctx:
        emb, attn = _prompt_embeds(janus, processor, img, device)          # [1,P,H],[1,P]
        P = emb.shape[1]
        tok_emb = janus.language_model.get_input_embeddings()(comp_ids.unsqueeze(0))  # [1,L,H]
        full = torch.cat([emb, tok_emb], dim=1)                            # [1,P+L,H]
        am = torch.cat([attn, torch.ones(1, comp_ids.shape[0], device=device, dtype=attn.dtype)], 1)
        pos = am.long().cumsum(-1) - 1
        pos.masked_fill_(am == 0, 0)
        logits = janus.language_model(inputs_embeds=full, attention_mask=am,
                                      position_ids=pos).logits[0]           # [P+L,V]
        # logits at position t predict token t+1; completion tokens live at P..P+L-1,
        # so their predictions come from logits at P-1 .. P+L-2.
        L = comp_ids.shape[0]
        pred = logits[P - 1: P - 1 + L]                                    # [L,V]
        logp = F.log_softmax(pred.float(), dim=-1)
        return logp.gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)          # [L]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    global USER_PROMPT                                      # reassigned from --user_prompt
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/Janus-Pro-1B")
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exclude_ids_json", default="", help="held-out ids to drop (eval split)")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--system_prompt", default=ANCHOR_SYSTEM_PROMPT,
                    help="fixed anchor system prompt (reusable identity); shared across all alpha")
    ap.add_argument("--user_prompt", default=USER_PROMPT,
                    help="fixed task+constraint user prompt; shared across all alpha")
    # LoRA (fresh, rank 8: the knob is ~1-D, low rank avoids memorising content)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    # GRPO / rollout
    ap.add_argument("--group_size", type=int, default=4, help="K completions per image")
    ap.add_argument("--images_per_step", type=int, default=2, help="grad-accum over N images")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)  # 1.0 degenerates this 1B base
    ap.add_argument("--top_p", type=float, default=0.9)         # nucleus truncation keeps rollouts coherent
    ap.add_argument("--beta_kl", type=float, default=0.04, help="KL-to-base weight")
    ap.add_argument("--w_rep", type=float, default=1.0, help="repetition-penalty weight in reward")
    ap.add_argument("--std_norm", action="store_true", help="divide advantages by group std (default: mean-only)")
    ap.add_argument("--l_verbose", type=float, default=R.L_VERBOSE_DEFAULT)
    ap.add_argument("--l_concise", type=float, default=R.L_CONCISE_DEFAULT)
    # optim
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--log_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)

    # ---- data ----
    rows = json.load(open(args.data_json))
    if args.exclude_ids_json:
        raw = json.load(open(args.exclude_ids_json))
        excl = {r["id"] if isinstance(r, dict) else r for r in raw}
        rows = [r for r in rows if r.get("id") not in excl]
    if args.max_samples:
        rows = rows[: args.max_samples]
    if is_main:
        print(f"[data] {len(rows)} images | K={args.group_size} "
              f"| anchors verbose={args.l_verbose} concise={args.l_concise}", flush=True)
        print(f"[anchor] system = {args.system_prompt!r}", flush=True)
        print(f"[anchor] user   = {args.user_prompt!r}", flush=True)

    # ---- model + processor ----
    USER_PROMPT = args.user_prompt                          # used by _prompt_embeds
    processor = VLChatProcessor.from_pretrained(args.model)
    processor.system_prompt = args.system_prompt            # fixed anchor identity
    processor.tokenizer = load_fast_tokenizer(args.model)   # spaces-preserving encode
    eos = processor.tokenizer.eos_token_id
    janus = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type=None)
    janus.language_model = get_peft_model(janus.language_model, peft_cfg)
    janus = janus.to(device).train()
    for n, p in janus.named_parameters():
        p.requires_grad = ("lora_" in n)
    # NOTE: gradient checkpointing corrupts this model's inputs_embeds generation loop
    # (produces gibberish), so we DON'T enable it globally. Instead it is toggled ON
    # only around the grad forward/backward via _set_gc() below, and OFF for all
    # generation / inference forwards. Non-reentrant so grads flow to the LoRA without
    # the enable_input_require_grads hook (which also corrupts the rollout).
    def _set_gc(on):
        if not args.gradient_checkpointing:
            return
        if on:
            janus.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            janus.language_model.gradient_checkpointing_disable()
    _set_gc(False)

    base_scaling = {}
    for name, m in janus.named_modules():
        if isinstance(m, LoraLayer):
            for adp in m.scaling:
                base_scaling[(name, adp)] = float(m.scaling[adp])
    trainable = [p for p in janus.parameters() if p.requires_grad]
    if is_main:
        print(f"[model] fresh rank-{args.lora_r} gran-LoRA | trainable "
              f"{sum(p.numel() for p in trainable):,}", flush=True)

    # ---- loader ----
    ds = ImageDataset(rows, args.data_dir)
    sampler = DistributedSampler(ds, shuffle=True) if world_size > 1 else None
    loader = DataLoader(ds, batch_size=args.images_per_step, sampler=sampler,
                        shuffle=(sampler is None), num_workers=4,
                        collate_fn=collate, drop_last=True)
    total_steps = args.max_steps if args.max_steps > 0 else len(loader) * args.epochs
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(args.warmup_ratio * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warmup) *
        (0.5 * (1 + math.cos(3.14159 * max(0, s - warmup) / max(1, total_steps - warmup)))))

    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            logs = {"reward": [], "r_len": [], "r_rep": [], "len": [], "alpha": [],
                    "kl": [], "adv_abs": []}
            sample_caps = []
            for b in batch:
                try:
                    img = Image.open(b["path"]).convert("RGB")
                except Exception:
                    continue
                alpha = torch.rand(1).item()          # sample alpha ~ U(0,1) per image

                # --- 1. rollout: K samples at scale=alpha (no grad) ---
                _set_gc(False)                        # gc corrupts generation -> off
                set_gran_scale(janus, base_scaling, alpha)
                comps = sample_group(janus, processor, img, device,
                                     k=args.group_size, max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature, top_p=args.top_p, eos=eos)
                if all(c.numel() == 0 for c in comps):
                    continue

                # --- 2. reward each completion ---
                rewards, infos, texts = [], [], []
                for c in comps:
                    text = processor.tokenizer.decode(c.tolist(), skip_special_tokens=True)
                    text = text.replace("Ġ", " ").replace("Ċ", "\n").strip()
                    tot, info = R.reward(text, alpha, c.numel(), w_rep=args.w_rep,
                                         l_verbose=args.l_verbose, l_concise=args.l_concise)
                    rewards.append(tot); infos.append(info); texts.append(text)
                rw = torch.tensor(rewards, device=device)

                # --- 3. GRPO advantage within the group (mean-only baseline) ---
                adv = rw - rw.mean()
                if args.std_norm:
                    adv = adv / (rw.std() + 1e-4)

                # --- 4. reference logprobs at scale 0 (= base, no grad) ---
                ref_lp = []
                set_gran_scale(janus, base_scaling, 0.0)
                for c in comps:
                    if c.numel() == 0:
                        ref_lp.append(None); continue
                    ref_lp.append(token_logprobs(janus, processor, img, c, device,
                                                 requires_grad=False).detach())

                # --- 5. policy logprobs at scale=alpha (grad) -> PG + KL loss ---
                # backward per-completion (only one graph alive at a time -> bounded
                # memory on 24GB); grads accumulate. denom = #non-empty * #images.
                n_used = sum(1 for c in comps if c.numel() > 0)
                if n_used == 0:
                    continue
                denom = n_used * len(batch)
                _set_gc(True)                         # gc on only for the grad forward
                set_gran_scale(janus, base_scaling, alpha)
                kl_acc = 0.0
                for c, a_i, rlp in zip(comps, adv, ref_lp):
                    if c.numel() == 0:
                        continue
                    lp = token_logprobs(janus, processor, img, c, device, requires_grad=True)  # [L]
                    # k3 KL estimator per token (>=0), ref is scale-0 base:
                    diff = rlp - lp
                    kl = (torch.exp(diff) - diff - 1.0)                    # [L]
                    pg = -a_i * lp                                         # [L]
                    tok_loss = (pg + args.beta_kl * kl).mean() / denom     # per-token mean
                    tok_loss.backward()
                    kl_acc += float(kl.mean().detach())

                logs["reward"] += rewards
                logs["r_len"] += [i["r_len"] for i in infos]
                logs["r_rep"] += [i["r_rep"] for i in infos]
                logs["len"] += [i["len_tok"] for i in infos]
                logs["alpha"].append(alpha)
                logs["kl"].append(kl_acc / n_used)
                logs["adv_abs"].append(float(adv.abs().mean()))
                sample_caps.append((alpha, texts[0]))

            # manual gradient all-reduce (we do multiple backwards + generation, so
            # we don't use DDP's forward wrapper; average lora grads across ranks).
            if world_size > 1:
                for p in trainable:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step(); sched.step(); step += 1

            if is_main and step % args.log_steps == 0 and logs["reward"]:
                m = lambda k: sum(logs[k]) / max(1, len(logs[k]))
                dt = time.perf_counter() - t0
                print(f"[step {step}/{total_steps}] r={m('reward'):+.3f} "
                      f"r_len={m('r_len'):+.3f} r_rep={m('r_rep'):+.3f} "
                      f"len={m('len'):.0f} kl={m('kl'):.3f} |adv|={m('adv_abs'):.3f} "
                      f"lr={sched.get_last_lr()[0]:.1e} ({step/max(dt,1e-3):.2f} it/s)", flush=True)
                a0, cap0 = sample_caps[-1]
                print(f"    sample a={a0:.2f}: {cap0[:160]}", flush=True)
            if is_main and step % args.save_steps == 0:
                set_gran_scale(janus, base_scaling, 1.0)
                janus.language_model.save_pretrained(out_dir)
                print(f"[save] adapter -> {out_dir} (step {step})", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    if is_main:
        set_gran_scale(janus, base_scaling, 1.0)
        janus.language_model.save_pretrained(out_dir)
        print(f"[done] saved gran-LoRA -> {out_dir}", flush=True)
    if world_size > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
