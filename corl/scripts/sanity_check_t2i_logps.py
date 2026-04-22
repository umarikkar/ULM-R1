"""
Sanity check for the off-by-one question in `_get_per_token_logps` (task='generation').

Procedure:
  1. Run a manual autoregressive t2i generation loop that records the CONDITIONAL
     raw logits used at each sampling step (no CFG, no temperature). These are
     the ground-truth "scoring" distribution: logits that actually sampled img_k.
  2. Re-score the same generated tokens via a single teacher-forcing forward pass
     over [prompt, img_0, ..., img_K-1], exactly as the trainer's
     `_get_per_token_logps` does.
  3. For each image-token position k, compute the log-prob the generation-time
     and re-scored distributions assign to the actual sampled token under two
     alignments:
       A) hidden_states[:, -K:, :]                 -- current trainer behavior
       B) hidden_states[:, P-1 : P-1+K, :]         -- shifted by -1

If alignment B matches the generation-time log-probs position-by-position and
alignment A is off by one index, the current code is off-by-one.

Run with: python corl/scripts/sanity_check_t2i_logps.py
"""

import os
import sys

import torch

torch.backends.cudnn.enabled = False  # match debug_corl_unified.py
os.environ.setdefault("WANDB_DISABLED", "true")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor


CKPT = "deepseek-ai/Janus-Pro-1B"
NUM_IMG_TOKENS = 16  # keep small for speed; alignment question doesn't depend on K


@torch.inference_mode()
def generate_and_record(model, processor, prompt_text, device, K=NUM_IMG_TOKENS):
    """Manual autoregressive loop: returns
        gen_tokens: [1, K] sampled image token ids
        step_logits: [K, V] raw conditional logits at each sampling step (no CFG)
    The conditional branch uses the real prompt (no pad mask).
    """
    tokenizer = processor.tokenizer
    prompt_ids = tokenizer([prompt_text], return_tensors="pt", padding=True).input_ids.to(device)
    attn_mask = torch.ones_like(prompt_ids)
    P = prompt_ids.size(1)

    # Embed prompt; start autoregressive loop
    inputs_embeds = model.language_model.get_input_embeddings()(prompt_ids)

    gen_tokens = torch.zeros((1, K), dtype=torch.long, device=device)
    step_logits = []
    past_kv = None
    running_attn = attn_mask.clone()

    for k in range(K):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=running_attn,
            use_cache=True,
            past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        hidden_last = outputs.last_hidden_state[:, -1, :]  # [1, D]
        logits = model.gen_head(hidden_last)  # [1, V]
        step_logits.append(logits.float().clone())

        # Argmax sample (deterministic so we can recompute identically)
        next_tok = logits.argmax(dim=-1, keepdim=True)  # [1, 1]
        gen_tokens[:, k] = next_tok.squeeze(-1)

        # Prepare next-step inputs_embeds
        img_embed = model.prepare_gen_img_embeds(next_tok.to(torch.long))  # [1, 1, D] maybe
        if img_embed.dim() == 2:
            img_embed = img_embed.unsqueeze(1)
        inputs_embeds = img_embed
        running_attn = torch.cat(
            [running_attn, torch.ones((1, 1), device=device, dtype=running_attn.dtype)], dim=1
        )

    step_logits = torch.cat(step_logits, dim=0)  # [K, V]
    return prompt_ids, attn_mask, gen_tokens, step_logits, P


@torch.inference_mode()
def score_full_sequence(model, prompt_ids, attn_mask, gen_tokens):
    """Teacher-forcing forward pass over [prompt, img_0..img_{K-1}].
    Returns the full hidden states, so we can slice under different alignments.
    """
    P = prompt_ids.size(1)
    K = gen_tokens.size(1)

    prompt_embed = model.language_model.get_input_embeddings()(prompt_ids)
    img_embed = model.prepare_gen_img_embeds(gen_tokens.to(torch.long))
    inputs_embeds = torch.cat([prompt_embed, img_embed], dim=1)
    full_attn = torch.cat(
        [attn_mask, torch.ones_like(gen_tokens, dtype=attn_mask.dtype)], dim=1
    )
    outputs = model.language_model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attn,
        use_cache=False,
        past_key_values=None,
    )
    hidden = outputs.last_hidden_state  # [1, P+K, D]
    assert hidden.size(1) == P + K
    return hidden, P, K


@torch.inference_mode()
def main():
    import math

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    processor = VLChatProcessor.from_pretrained(CKPT)
    processor.system_prompt = ""
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    convo = [
        {"role": "<|User|>", "content": "A photo of a red apple on a wooden table."},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=convo, sft_format=processor.sft_format, system_prompt="",
    )
    prompt_text = sft + processor.image_start_tag
    print(f"prompt_text (repr, first 120 chars): {repr(prompt_text)[:120]}")

    # 1) Generate + record step logits (no CFG)
    prompt_ids, attn_mask, gen_tokens, step_logits, P = generate_and_record(
        model, processor, prompt_text, device, K=NUM_IMG_TOKENS
    )
    print(f"P (prompt len) = {P}, K (img tokens) = {gen_tokens.size(1)}")
    print(f"first 10 generated tokens: {gen_tokens[0, :10].tolist()}")

    # 2) Teacher-forcing forward, keep full hidden states
    hidden, P2, K = score_full_sequence(model, prompt_ids, attn_mask, gen_tokens)
    assert P2 == P

    # 3) Compute logits under both alignments
    logits_A = model.gen_head(hidden[:, -K:, :]).float()              # positions [P, P+K-1]
    logits_B = model.gen_head(hidden[:, P - 1 : P - 1 + K, :]).float() # positions [P-1, P+K-2]

    # 3b) Also score through model.forward(task="generation") — i.e. the actual
    # code path used by the trainer's _get_per_token_logps. After the fix this
    # should match the generation-time logits (= alignment B).
    fwd_out = model(
        t2i_input_ids=prompt_ids,
        t2i_attention_mask=attn_mask,
        t2i_discrete_img_ids=gen_tokens,
        t2i_logits_to_keep=K,
        task="generation",
    )
    logits_fwd = fwd_out.logits.float()

    V = logits_A.size(-1)
    print(f"image vocab V = {V}   uniform log-prob = {-math.log(V):.3f}")

    # Log-prob of the actually-generated tokens under each alignment
    def lp_of(logits, ids):
        return torch.log_softmax(logits, dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)

    lp_gen = lp_of(step_logits.unsqueeze(0), gen_tokens)[0]  # [K]
    lp_A = lp_of(logits_A, gen_tokens)[0]
    lp_B = lp_of(logits_B, gen_tokens)[0]

    # ---- Summaries ----
    def describe(name, x):
        print(f"  {name:30s} mean={x.mean().item():.4f}  median={x.median().item():.4f}  "
              f"min={x.min().item():.4f}  max={x.max().item():.4f}")

    print("\nLog-probs of sampled (argmax) tokens:")
    describe("generation-time (truth)", lp_gen)
    describe("A: hidden[:, -K:]       ", lp_A)
    describe("B: hidden[:, P-1:P-1+K] ", lp_B)

    # ---- Position-by-position comparison ----
    # The correct alignment should match lp_gen almost exactly (floating point only).
    print("\nPer-position log-probs (first 8 positions):")
    print(f"{'k':>3}  {'tok':>6}  {'lp_gen':>9}  {'lp_A':>9}  {'lp_B':>9}  {'A-gen':>9}  {'B-gen':>9}")
    for k in range(min(8, K)):
        print(
            f"{k:>3}  {gen_tokens[0, k].item():>6}  "
            f"{lp_gen[k].item():>9.4f}  {lp_A[k].item():>9.4f}  {lp_B[k].item():>9.4f}  "
            f"{(lp_A[k] - lp_gen[k]).item():>9.4f}  {(lp_B[k] - lp_gen[k]).item():>9.4f}"
        )

    # ---- Direct logits equality check ----
    # If alignment B is correct, logits_B[:, k] should equal step_logits[k] up to fp noise.
    # If alignment A is correct, logits_A[:, k] should equal step_logits[k].
    diff_A = (logits_A[0] - step_logits).abs().mean(dim=-1)  # [K]
    diff_B = (logits_B[0] - step_logits).abs().mean(dim=-1)  # [K]
    print("\nMean |logits - step_logits| per position (first 8):")
    print(f"{'k':>3}  {'|A - gen|':>12}  {'|B - gen|':>12}")
    for k in range(min(8, K)):
        print(f"{k:>3}  {diff_A[k].item():>12.5f}  {diff_B[k].item():>12.5f}")

    print(f"\nOverall mean |A - gen| = {diff_A.mean().item():.5f}")
    print(f"Overall mean |B - gen| = {diff_B.mean().item():.5f}")

    # ---- Check the actual model.forward(task="generation") output ----
    diff_fwd = (logits_fwd[0] - step_logits).abs().mean(dim=-1)
    print(f"Overall mean |model.forward(task='generation') - gen| = {diff_fwd.mean().item():.5f}")
    print("(After the fix, this should match |B - gen|, not |A - gen|.)")

    # ---- Off-by-one signature check ----
    # If alignment A is shifted by +1 relative to the truth, then for each k < K-1
    # we should see logits_A[:, k] == logits_B[:, k+1] up to fp noise.
    if K >= 2:
        shift_diff = (logits_A[0, : K - 1] - logits_B[0, 1:K]).abs().mean().item()
        print(f"Shift check: mean |logits_A[:,:-1] - logits_B[:,1:]| = {shift_diff:.5f}")
        print("(Near-zero here means alignment A is exactly alignment B shifted by +1.)")

    # ---- Verdict ----
    # bf16 on full [V]-dim logits has much larger fp noise than on log-probs of a
    # single sampled token, so use different tolerances for each.
    tol = 0.15
    B_matches = diff_B.mean().item() < tol
    A_matches = diff_A.mean().item() < tol
    print("\nVerdict:")
    if B_matches and not A_matches:
        print("  Alignment B (hidden[:, P-1:P-1+K]) matches generation logits.")
        print("  Alignment A (hidden[:, -K:]) does NOT.")
        print("  => The trainer's current slicing is off-by-one: it is scoring img_k")
        print("     with the logits that would predict img_{k+1}.")
    elif A_matches and not B_matches:
        print("  Alignment A matches generation logits — current slicing is correct.")
    elif A_matches and B_matches:
        print("  Both match — unlikely; investigate.")
    else:
        print("  Neither matches exactly — investigate (could be dtype/precision).")


if __name__ == "__main__":
    main()
