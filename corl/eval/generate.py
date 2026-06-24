"""Generate T2I outputs for every row in test_split.json from a checkpoint.

Stage 1 of the eval pipeline. Saves one PNG per test row plus a manifest.json
that downstream metric scripts consume. Sharding support so multiple GPUs can
process disjoint slices in parallel.

Usage:
    # Vanilla (no fine-tune): just point at the base model id.
    python corl/eval/generate.py \\
        --base_model deepseek-ai/Janus-Pro-1B \\
        --test_split corl/eval/test_split.json \\
        --data_dir /work/.../PubMedVision \\
        --out_dir results/eval/vanilla

    # Fine-tuned checkpoint (LoRA + modules_to_save dir):
    python corl/eval/generate.py \\
        --base_model deepseek-ai/Janus-Pro-1B \\
        --adapter_dir results/.../checkpoint-12000 \\
        --test_split corl/eval/test_split.json \\
        --data_dir /work/.../PubMedVision \\
        --out_dir results/eval/exp5

    # 8-way sharding across GPUs (run each on a different CUDA_VISIBLE_DEVICES):
    SHARD=0 NUM_SHARDS=8 python corl/eval/generate.py ...

If the checkpoint has `text_to_proto` and `prototype_emb` attached, cond is
built from the caption and additively biased onto every image-token embedding
(matching _generate_one_image in the trainer).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor

torch.backends.cudnn.enabled = False

N_IMAGE_TOKENS = 576
IMG_SIZE = 384
PATCH_SIZE = 16

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_biomedclip(device):
    import open_clip
    m, _ = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    m = m.visual.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    try:
        res = int(m.image_size if isinstance(m.image_size, int) else m.image_size[0])
    except Exception:
        res = 224
    return m, res


@torch.inference_mode()
def image_to_proto_w(image_path: str, bmc, bmc_res: int,
                    centroids: torch.Tensor, temperature: float, device):
    img = Image.open(image_path).convert("RGB").resize(
        (bmc_res, bmc_res), Image.Resampling.BICUBIC
    )
    arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    arr = arr.unsqueeze(0).to(device)
    arr = (arr - CLIP_MEAN.to(device)) / CLIP_STD.to(device)
    feat = bmc(arr.to(dtype=torch.bfloat16)).float()
    feat = F.normalize(feat, dim=-1)
    c = centroids.to(device=device, dtype=torch.float32)
    c = c / (c.norm(dim=-1, keepdim=True) + 1e-8)
    sims = feat @ c.t()
    return F.softmax(sims / max(temperature, 1e-6), dim=-1).squeeze(0)


def load_model(base_model: str, adapter_dir: str | None, device):
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    processor = VLChatProcessor.from_pretrained(base_model)
    processor.system_prompt = ""

    if adapter_dir:
        # PEFT-style adapter. Imports kept local to avoid hard dep when vanilla.
        from peft import PeftModel
        # Attach prototype_emb / text_to_proto before loading the adapter so the
        # state_dict can populate them. We discover K from the adapter config.
        adapter_cfg_path = Path(adapter_dir) / "adapter_config.json"
        mts = []
        if adapter_cfg_path.exists():
            mts = json.load(open(adapter_cfg_path)).get("modules_to_save") or []
        if "prototype_emb" in mts or _has_proto_in_state(adapter_dir):
            K, d_feat = _peek_K_d(adapter_dir)
            d_model = model.language_model.config.hidden_size
            model.prototype_emb = torch.nn.Embedding(K + 1, d_model, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, adapter_dir)
        # text_to_proto is attached after PEFT wrapping and saved separately by
        # TextToProtoSaveCallback as text_to_proto.safetensors next to the adapter.
        t2p_path = Path(adapter_dir) / "text_to_proto.safetensors"
        if t2p_path.exists() and hasattr(model.base_model.model, "prototype_emb"):
            from safetensors.torch import load_file
            inner = model.base_model.model
            d_model = inner.language_model.config.hidden_size
            K = inner.prototype_emb.num_embeddings - 1
            inner.text_to_proto = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_model, dtype=torch.bfloat16),
                torch.nn.GELU(),
                torch.nn.Linear(d_model, K, dtype=torch.bfloat16),
            )
            sd = load_file(str(t2p_path))
            missing, unexpected = inner.text_to_proto.load_state_dict(sd, strict=False)
            print(f"[gen] text_to_proto loaded: missing={len(missing)} unexpected={len(unexpected)}")
            inner.text_to_proto.to(device)
        # Merge LoRA deltas into the base linear layers so inference runs at
        # vanilla speed (no extra per-layer matmul). prototype_emb and
        # text_to_proto survive because they're modules_to_save / attached
        # submodules respectively, not LoRA adapters.
        model = model.merge_and_unload()
        # prototype_emb was instantiated on CPU before PEFT load and
        # merge_and_unload doesn't guarantee device consistency for
        # non-LoRA submodules. Force the whole merged model to device.
        model = model.to(device)
        print("[gen] LoRA merged into base; PEFT wrapper removed; model on", device)
        print(f"[gen] loaded adapter from {adapter_dir} "
              f"(prototype_emb={hasattr(model, 'prototype_emb')}, "
              f"text_to_proto={_inner_has_attr(model, 'text_to_proto')})")
    return model, processor


def _has_proto_in_state(adapter_dir):
    return _state_has_prefix(adapter_dir, "prototype_emb")


def _has_text_to_proto_in_state(adapter_dir):
    return _state_has_prefix(adapter_dir, "text_to_proto")


def _state_has_prefix(adapter_dir, prefix):
    sf = Path(adapter_dir) / "adapter_model.safetensors"
    if sf.exists():
        from safetensors import safe_open
        with safe_open(str(sf), framework="pt") as f:
            return any(prefix in k for k in f.keys())
    bn = Path(adapter_dir) / "adapter_model.bin"
    if bn.exists():
        state = torch.load(str(bn), map_location="cpu")
        return any(prefix in k for k in state.keys())
    return False


def _peek_K_d(adapter_dir):
    """Read prototype_emb shape from the adapter state to infer K, d_feat."""
    sf = Path(adapter_dir) / "adapter_model.safetensors"
    if sf.exists():
        from safetensors import safe_open
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                if "prototype_emb" in k and "weight" in k:
                    t = f.get_tensor(k)
                    return t.shape[0] - 1, t.shape[1]
    raise RuntimeError("prototype_emb weight not found in adapter")


def _load_text_to_proto_state(adapter_dir, inner_model):
    sf = Path(adapter_dir) / "adapter_model.safetensors"
    if sf.exists():
        from safetensors.torch import load_file
        state = load_file(str(sf))
    else:
        state = torch.load(str(Path(adapter_dir) / "adapter_model.bin"), map_location="cpu")
    sub = {}
    for k, v in state.items():
        if "text_to_proto" in k:
            new_k = k.split("text_to_proto.")[-1]
            sub[new_k] = v
    if sub:
        missing, unexpected = inner_model.text_to_proto.load_state_dict(sub, strict=False)
        print(f"[gen] text_to_proto loaded: missing={len(missing)} unexpected={len(unexpected)}")


def _inner_has_attr(model, name):
    if hasattr(model, name):
        return True
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return hasattr(model.base_model.model, name)
    return False


def _inner(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


@torch.inference_mode()
def generate_image(model, processor, caption: str, device,
                   cfg_scale=5.0, temperature=1.0, parallel_size=1,
                   centroids: torch.Tensor | None = None,
                   image_w_proto: torch.Tensor | None = None):
    """If `image_w_proto` is provided ([K] soft-assignment over prototypes
    derived from the GT image's BiomedCLIP feature), it overrides the text
    head and is used directly to build the prototype bias."""
    inner = _inner(model)
    has_proto = hasattr(inner, "prototype_emb") and (
        image_w_proto is not None or hasattr(inner, "text_to_proto")
    )

    conv = [
        {"role": "<|User|>", "content": caption},
        {"role": "<|Assistant|>", "content": ""},
    ]
    prompt = (
        processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conv, sft_format=processor.sft_format, system_prompt="",
        )
        + processor.image_start_tag
    )
    enc = processor.tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    pad_id = processor.pad_id
    bos_id = processor.tokenizer.bos_token_id
    L = input_ids.shape[1]

    tokens = torch.zeros((parallel_size * 2, L), dtype=torch.long, device=device)
    attn = torch.zeros((parallel_size * 2, L), dtype=attention_mask.dtype, device=device)
    for i in range(parallel_size * 2):
        tokens[i] = input_ids[0]
        attn[i] = attention_mask[0]
        if i % 2 != 0:
            bp = (tokens[i] == bos_id).nonzero(as_tuple=True)[0]
            if len(bp) > 0:
                tokens[i, bp[0].item() + 1:-1] = pad_id

    inputs_embeds = inner.language_model.get_input_embeddings()(tokens)

    # Build prototype bias: either from the GT image's BiomedCLIP soft-assignment
    # (oracle / upper-bound) or from the caption via the text_to_proto head.
    proto_bias_2x = None
    if has_proto:
        proto_w = inner.prototype_emb.weight
        if image_w_proto is not None:
            w = image_w_proto.to(device=device, dtype=torch.float32).view(1, -1)
        else:
            cond_in = inner.language_model.get_input_embeddings()(input_ids)
            out = inner.language_model.model(inputs_embeds=cond_in, attention_mask=attention_mask)
            h = out.last_hidden_state
            m = attention_mask.to(dtype=h.dtype)
            pooled = (h * m.unsqueeze(-1)).sum(dim=1) / m.sum(dim=1, keepdim=True).clamp(min=1.0)
            head_dtype = next(inner.text_to_proto.parameters()).dtype
            text_logits = inner.text_to_proto(pooled.to(dtype=head_dtype))
            w = F.softmax(text_logits.float(), dim=-1)
        cond_vec = w.to(proto_w.dtype) @ proto_w[1:]
        uncond_vec = proto_w[0:1].to(cond_vec.dtype)
        d_h = cond_vec.shape[-1]
        proto_bias_2x = torch.zeros(parallel_size * 2, d_h, device=device, dtype=cond_vec.dtype)
        proto_bias_2x[0::2] = cond_vec.expand(parallel_size, -1)
        proto_bias_2x[1::2] = uncond_vec.expand(parallel_size, -1)

    generated_tokens = torch.zeros((parallel_size, N_IMAGE_TOKENS), dtype=torch.long, device=device)
    outputs = None
    for i in range(N_IMAGE_TOKENS):
        outputs = inner.language_model.model(
            inputs_embeds=inputs_embeds, attention_mask=attn,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
        )
        hidden = outputs.last_hidden_state[:, -1, :]
        logits = inner.gen_head(hidden)
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_scale * (logit_cond - logit_uncond)
        probs = torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        generated_tokens[:, i] = next_token
        both = torch.stack([next_token, next_token], dim=1).view(-1)
        inputs_embeds = inner.prepare_gen_img_embeds(both).unsqueeze(1)
        if proto_bias_2x is not None:
            inputs_embeds = inputs_embeds + proto_bias_2x.to(inputs_embeds.dtype).unsqueeze(1)
        attn = torch.cat(
            [attn, torch.ones(attn.shape[0], 1, dtype=attn.dtype, device=device)], dim=1,
        )

    grid = IMG_SIZE // PATCH_SIZE
    dec = inner.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, grid, grid],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(dec[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="deepseek-ai/Janus-Pro-1B")
    ap.add_argument("--adapter_dir", default="")
    ap.add_argument("--test_split", required=True)
    ap.add_argument("--data_dir", required=True,
                    help="Base dir for image paths in test_split.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--caption_field", default="Original_Caption",
                    help="Which caption to use as the prompt.")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", "0")))
    ap.add_argument("--num_shards", type=int, default=int(os.environ.get("NUM_SHARDS", "1")))
    ap.add_argument("--proto_source", choices=["text", "image"], default="text",
                    help="Build prototype cond from text_to_proto(caption) (default) "
                         "or from BiomedCLIP(GT image) soft-assignment (oracle/UB).")
    ap.add_argument("--prototype_centroids_path", default="",
                    help="Required when --proto_source=image. BiomedCLIP feature centroids .pt.")
    ap.add_argument("--cond_temperature", type=float, default=0.1,
                    help="Softmax temperature for image->proto soft assignment.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    with open(args.test_split) as f:
        rows = json.load(f)
    # Shard by index modulo num_shards so coverage stays per-modality.
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    print(f"[gen] shard {args.shard}/{args.num_shards}: {len(rows)} rows")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(args.base_model,
                                  args.adapter_dir or None, device)

    bmc, bmc_res, centroids = None, None, None
    if args.proto_source == "image":
        if not args.prototype_centroids_path:
            raise SystemExit("--prototype_centroids_path is required when --proto_source=image")
        cd = torch.load(args.prototype_centroids_path, map_location="cpu")
        centroids = cd["centroids"].float()
        bmc, bmc_res = load_biomedclip(device)
        bmc = bmc.to(dtype=torch.bfloat16)
        print(f"[gen] proto_source=image  centroids K={cd.get('K', centroids.shape[0])} "
              f"d_feat={centroids.shape[-1]} tau={args.cond_temperature}")

    manifest_path = out_dir / f"manifest_shard{args.shard}.json"
    manifest = []
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        out_png = out_dir / "images" / f"{r['id']}.png"
        if out_png.exists():
            # Resume.
            manifest.append({**r, "gen_path": str(out_png)})
            continue
        cap = r.get(args.caption_field) or r.get("Original_Caption")
        if isinstance(cap, list):
            cap = cap[0]
        try:
            w_img = None
            if args.proto_source == "image":
                gt_path = os.path.join(args.data_dir, r["image"])
                w_img = image_to_proto_w(gt_path, bmc, bmc_res, centroids,
                                          args.cond_temperature, device)
            img = generate_image(model, processor, cap, device,
                                 cfg_scale=args.cfg_scale,
                                 temperature=args.temperature,
                                 image_w_proto=w_img)
            img.save(out_png)
            manifest.append({**r, "gen_path": str(out_png)})
        except Exception as e:
            print(f"[gen] {r['id']} failed: {e}")
            continue
        if (i + 1) % 50 == 0:
            dt = time.perf_counter() - t0
            print(f"[gen] {i+1}/{len(rows)} done ({(i+1)/dt:.2f} img/s)")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"[gen] wrote {len(manifest)} rows -> {manifest_path}")


if __name__ == "__main__":
    main()
