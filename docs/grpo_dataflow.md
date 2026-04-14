# GRPO Unified Training — Data Flow

## Overview

```
Dataset (text prompts, images, QA pairs)
                │
                ▼
┌───────────────────────────────────┐
│  _generate_and_score_completions  │
│  (all NO GRAD)                    │
└───────────────────────────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
   ┌─────────┐     ┌──────────┐
   │  T2I    │     │  MM2T    │
   │ Generate│     │ Generate │
   └─────────┘     └──────────┘
        │                │
        ▼                ▼
  codebook IDs      text token IDs
  [bs*G, 576]       [bs*G, C]
        │                │
        ▼                ▼
   ┌─────────┐     ┌──────────┐
   │  T2I    │     │  MM2T    │
   │ Rewards │     │ Rewards  │
   └─────────┘     └──────────┘
        │                │
        └───────┬────────┘
                ▼
         ┌────────────┐
         │ Advantages │  (normalized rewards per group)
         └────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  compute_loss                     │
│  (WITH GRAD)                      │
└───────────────────────────────────┘
                │
                ▼
          GRPO Policy Loss
                │
                ▼
           backward()
```

## Phase 1: Generation (no gradients)

```
┌─────────────────────────── T2I Generation ───────────────────────────┐
│                                                                      │
│  text prompt ──► tokenizer ──► t2i_prompt_ids [bs, P]                │
│                                       │                              │
│                                       ▼                              │
│                           t2i_generate_parallel()                    │
│                          (@torch.inference_mode)                     │
│                                       │                              │
│                    ┌──────────────────────────────────┐               │
│                    │  for i in range(576):            │               │
│                    │    LM backbone ──► gen_head      │               │
│                    │         │                        │               │
│                    │         ▼                        │               │
│                    │    logits over codebook          │               │
│                    │         │                        │               │
│                    │         ▼                        │               │
│                    │    CFG: uncond + w*(cond-uncond) │               │
│                    │         │                        │               │
│                    │         ▼                        │               │
│                    │    sample codebook index         │               │
│                    │         │                        │               │
│                    │         ▼                        │               │
│                    │    codebook lookup ──► embedding │               │
│                    │    (feed back as next input)     │               │
│                    └──────────────────────────────────┘               │
│                                       │                              │
│                                       ▼                              │
│                    t2i_completion_ids [bs*G, 576]  (codebook indices) │
│                    t2i_completions    (decoded PIL images)            │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── MM2T Generation ──────────────────────────┐
│                                                                      │
│  image + QA prompt ──► processor ──► mm2t_prompt_ids [bs, P]         │
│                                      pixel_values    [bs, 1, 3, H, W]│
│                                             │                        │
│                                             ▼                        │
│                              prepare_inputs_embeds()                 │
│                    ┌──────────────────────────────────┐               │
│                    │  text tokens ──► token embeddings│               │
│                    │  pixel_values ──► SigLIP encoder │               │
│                    │       ──► image patch embeddings │               │
│                    │  replace placeholders with       │               │
│                    │       image embeddings           │               │
│                    └──────────────────────────────────┘               │
│                                             │                        │
│                                             ▼                        │
│                              inputs_embeds [bs, P, 2048]             │
│                                             │                        │
│                                             ▼                        │
│                              language_model.generate()               │
│                              (autoregressive text sampling)          │
│                                             │                        │
│                                             ▼                        │
│                    mm2t_completion_ids [bs*G, C]  (text token IDs)    │
│                    mm2t_completions    (decoded strings)              │
└──────────────────────────────────────────────────────────────────────┘
```

## Phase 2: Reward & Advantage Computation (no gradients)

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  T2I rewards:                                                        │
│    t2i_completions (PIL images) + prompts                            │
│         │                                                            │
│         ├──► cycle consistency reward (generate──►understand──►match) │
│         ├──► text-image similarity reward                            │
│         └──► ... other reward functions                              │
│         │                                                            │
│         ▼                                                            │
│    t2i_rewards [bs*G]                                                │
│                                                                      │
│  MM2T rewards:                                                       │
│    mm2t_completions (text strings) + QA ground truth                 │
│         │                                                            │
│         ├──► QA accuracy reward                                      │
│         ├──► format reward                                           │
│         └──► ... other reward functions                              │
│         │                                                            │
│         ▼                                                            │
│    mm2t_rewards [bs*G]                                               │
│                                                                      │
│  Advantage computation:                                              │
│    if unify_reward:                                                  │
│         unified = t2i_rewards + 0.8 * mm2t_rewards                   │
│         advantage = (unified - mean) / (std + eps)                   │
│    else:                                                             │
│         t2i_advantage  = (t2i_rewards  - mean) / (std + eps)         │
│         mm2t_advantage = (mm2t_rewards - mean) / (std + eps)         │
│              (grouped per prompt, across G generations)               │
└──────────────────────────────────────────────────────────────────────┘
```

## Phase 3: compute_loss (WITH gradients)

```
┌──────────────── model.forward(task='unify') ─────────────────────────┐
│                                                                      │
│  ┌─── MM2T branch ──────────────────────────────────┐                │
│  │                                                   │                │
│  │  mm2t_prompt_completion_ids [bs*G, P+C]           │                │
│  │  pixel_values [bs*G, 1, 3, H, W]                 │                │
│  │         │                                         │                │
│  │         ▼                                         │                │
│  │  prepare_inputs_embeds()                          │                │
│  │  (text embed + SigLIP image embed)                │                │
│  │         │                                         │                │
│  │         ▼                                         │                │
│  │  language_model() ──► mm2t_logits [bs*G, C, V]    │                │
│  │         │                                         │                │
│  │         ▼                                         │                │
│  │  selective_log_softmax(logits, token_ids)          │                │
│  │         │                                         │                │
│  │         ▼                                         │                │
│  │  mm2t_per_token_logps [bs*G, C]  ◄── HAS GRAD    │                │
│  └───────────────────────────────────────────────────┘                │
│                                                                      │
│  ┌─── T2I branch ───────────────────────────────────┐                │
│  │                                                   │                │
│  │  t2i_prompt_ids [bs*G, P]                         │                │
│  │  t2i_completion_ids [bs*G, 576]  (codebook IDs)   │                │
│  │         │                                         │                │
│  │         ▼                                         │                │
│  │  token embed(prompt) + codebook lookup(completion)│                │
│  │  concat ──► [bs*G, P+576, 2048]                   │                │
│  │         │                                         │                │
│  │         ▼                                         │                │
│  │  language_model.model() ──► hidden_states         │                │
│  │         │                                         │                │
│  │         ▼                                         │                │
│  │  gen_head() ──► t2i_logits [bs*G, 576, codebook]  │                │
│  │         │                                         │                │
│  │         ▼                                         │                │
│  │  selective_log_softmax(logits, codebook_ids)       │                │
│  │         │                                         │                │
│  │         ▼                                         │                │
│  │  t2i_per_token_logps [bs*G, 576]  ◄── HAS GRAD   │                │
│  └───────────────────────────────────────────────────┘                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────── GRPO Loss ───────────────────────────────────────────┐
│                                                                      │
│  ratio = exp(per_token_logps - old_per_token_logps)  ◄── HAS GRAD   │
│  advantages                                          ◄── NO GRAD    │
│                                                                      │
│  clipped_ratio = clamp(ratio, 1-eps, 1+eps)                          │
│  loss = -min(ratio * advantage, clipped_ratio * advantage)           │
│                                                                      │
│  + optional KL penalty (beta * KL divergence from ref model)         │
│                                                                      │
│         │                                                            │
│         ▼                                                            │
│    loss.backward()  ──► gradients flow back through LM               │
└──────────────────────────────────────────────────────────────────────┘
```

## Key: What has gradients and what doesn't

| Component | Gradients? | Why |
|-----------|-----------|-----|
| `t2i_generate_parallel` | No | `@torch.inference_mode()` — just sampling |
| `language_model.generate` | No | HF generate uses `torch.no_grad` internally |
| Reward functions | No | External scoring, no model params involved |
| Advantages | No | Derived from rewards, detached scalars |
| `old_per_token_logps` | No | Computed under `torch.inference_mode()` |
| `ref_per_token_logps` | No | Computed under `torch.inference_mode()` |
| **`per_token_logps` in `compute_loss`** | **Yes** | **This is the only place gradients are tracked** |
