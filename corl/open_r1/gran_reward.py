"""Reward for gran-LoRA RL: a length-target knob + a repetition regularizer.

Design (see HANDOFF_GRANULARITY_RL.md, revised):
  * The gran-LoRA scale alpha in [0,1] is the granularity knob. We do NOT supervise
    exact caption text (that memorises content); we reward a *sampled* caption for
    hitting a TARGET LENGTH set by alpha. alpha does content-selection at the
    generation level; length is the observable we shape.
  * Semantics (chosen): alpha=1 -> concise (short, ~l1), alpha=0 -> verbose (~l3).
  * Length is interpolated GEOMETRICALLY (log-linear) between the two anchors,
    because the measured levels are near-geometrically spaced in tokens
    (l1~23, l2~55, l3~137; sqrt(23*137)=56 ~= l2). Equal alpha steps => equal ratios.
  * Reward term 1 (drives alpha), in log-length so it is scale-free/symmetric:
        r_len = - | log(len_tok(c)) - log_target(alpha) |
  * Reward term 2 (regularizer only, can never help): a repetition penalty
        r_rep = - rep_frac_n(c)      (1 - distinct-n; heavy looping -> ~ -1)
  * Faithfulness is NOT in the reward. It is held by the KL-to-base anchor in the
    RL loop. These two terms catch dumb (repetition) padding, not smart
    (hallucinated) padding -- log sample captions to catch drift.

Anchors default to the medians measured on test_split_levels.json (fast tokenizer).
Recompute with `python -m corl.open_r1.gran_reward --validate ...`.
"""
import math
import re
from collections import Counter

# median token lengths (fast tokenizer) of cached l1 / l3 -> the alpha endpoints.
L_CONCISE_DEFAULT = 23.0    # alpha = 1
L_VERBOSE_DEFAULT = 137.0   # alpha = 0


def log_target(alpha, l_verbose=L_VERBOSE_DEFAULT, l_concise=L_CONCISE_DEFAULT):
    """log target token-length for a given alpha in [0,1] (geometric interp)."""
    return (1.0 - alpha) * math.log(l_verbose) + alpha * math.log(l_concise)


def target_len(alpha, l_verbose=L_VERBOSE_DEFAULT, l_concise=L_CONCISE_DEFAULT):
    """target token-length for alpha (for logging/inspection)."""
    return math.exp(log_target(alpha, l_verbose, l_concise))


_WORD = re.compile(r"\w+")


def _tokens_for_rep(text):
    """lightweight word tokens for the repetition statistic (tokenizer-free)."""
    return _WORD.findall(text.lower())


def rep_frac(text, n=3):
    """Repetition fraction = 1 - distinct-n over word n-grams.

    0.0 = every n-gram unique; -> 1.0 = degenerate looping. Tokenizer-free so the
    penalty is cheap and identical on CPU/GPU. For very short captions (< n words)
    returns 0 (nothing to repeat)."""
    toks = _tokens_for_rep(text)
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    distinct = len(set(grams)) / len(grams)
    return 1.0 - distinct


def length_reward(len_tok, alpha, l_verbose=L_VERBOSE_DEFAULT, l_concise=L_CONCISE_DEFAULT):
    """-|log len - log_target(alpha)|; scale-free, symmetric, 0 at target."""
    return -abs(math.log(max(int(len_tok), 1)) - log_target(alpha, l_verbose, l_concise))


def reward(text, alpha, len_tok, *, w_rep=1.0, rep_n=3,
           l_verbose=L_VERBOSE_DEFAULT, l_concise=L_CONCISE_DEFAULT):
    """Total reward and its breakdown.

    text     : decoded caption (for the repetition stat).
    alpha    : granularity knob in [0,1] used to sample this caption.
    len_tok  : #generated tokens (count in the SAME tokenizer the sampler uses).
    w_rep    : weight on the repetition penalty (>=0; penalty only).
    Returns (total: float, info: dict).
    """
    r_len = length_reward(len_tok, alpha, l_verbose, l_concise)
    r_rep = -rep_frac(text, rep_n)
    total = r_len + w_rep * r_rep
    return total, {"len_tok": int(len_tok), "r_len": r_len, "r_rep": r_rep,
                   "target_len": target_len(alpha, l_verbose, l_concise)}


# --------------------------------------------------------------------------- #
# Validation harness: prove the reward is (a) level-appropriate and (b) not
# gameable by length/repetition padding.  CPU-only.
# --------------------------------------------------------------------------- #
def _validate(args):
    import json
    import statistics as st
    import sys
    sys.path.insert(0, ".")
    from corl.open_r1.janus_tokenizer_fix import load_fast_tokenizer

    tok = load_fast_tokenizer(args.model)
    tlen = lambda s: len(tok.encode(s))
    rows = json.load(open(args.data_json))[: args.n]

    # 1) recompute anchors from this split (median token length of l1 / l3)
    med = {}
    for lv in ("l1", "l2", "l3"):
        t = [tlen(r[f"cached_captions_{lv}"]) for r in rows]
        med[lv] = st.median(t)
    l_concise, l_verbose = med["l1"], med["l3"]
    print(f"[anchors] l1(concise,a=1)={l_concise:.0f}  l2={med['l2']:.0f}  "
          f"l3(verbose,a=0)={l_verbose:.0f}  geom_mid={math.sqrt(l_concise*l_verbose):.0f}")

    # the alpha each cached level corresponds to under our semantics
    level_alpha = {"l3": 0.0, "l2": 0.5, "l1": 1.0}

    # 2) level-appropriateness: for each cached level, reward should PEAK at the
    #    alpha whose target matches that level (argmax over an alpha grid).
    grid = [i / 10 for i in range(11)]
    print("\n[peak] mean length-reward over cached captions, per (true level x alpha):")
    print("        " + "".join(f"a={a:>3.1f} " for a in grid))
    ok_peak = True
    for lv in ("l3", "l2", "l1"):
        caps = [r[f"cached_captions_{lv}"] for r in rows]
        lens = [tlen(c) for c in caps]
        means = [st.mean([length_reward(L, a, l_verbose, l_concise) for L in lens]) for a in grid]
        amax = grid[max(range(len(grid)), key=lambda i: means[i])]
        hit = abs(amax - level_alpha[lv]) <= 0.1
        ok_peak &= hit
        print(f"  {lv} " + "".join(f"{m:>6.2f}" for m in means) +
              f"   argmax a={amax:.1f} (want {level_alpha[lv]:.1f}) {'OK' if hit else 'MISS'}")

    # 3) hackability: take real l1 captions (target for alpha=1) and try to pad
    #    them up to l3 length. A good reward must NOT reward this at alpha=1.
    l1s = [r["cached_captions_l1"] for r in rows[: args.n_hack]]

    def pad_repeat(s):  # dumb padding: repeat the sentence to ~l3 length
        out = s
        while tlen(out) < l_verbose:
            out = out + " " + s
        return out

    def pad_filler(s):  # smart-ish padding: distinct filler words (no repeats)
        fillers = ("moreover additionally furthermore notably importantly consequently "
                   "subsequently accordingly nevertheless meanwhile ultimately specifically "
                   "particularly essentially fundamentally generally typically frequently "
                   "occasionally significantly").split()
        out, k = s, 0
        while tlen(out) < l_verbose:
            out = out + " " + fillers[k % len(fillers)]
            k += 1
        return out

    a = 1.0  # concise target: padded captions should score WORSE than the clean l1
    base_r = st.mean([reward(s, a, tlen(s), w_rep=args.w_rep)[0] for s in l1s])
    rep_r = st.mean([reward(pad_repeat(s), a, tlen(pad_repeat(s)), w_rep=args.w_rep)[0] for s in l1s])
    fil_r = st.mean([reward(pad_filler(s), a, tlen(pad_filler(s)), w_rep=args.w_rep)[0] for s in l1s])
    print(f"\n[hack] at alpha=1 (concise target), mean total reward:")
    print(f"  clean l1        : {base_r:+.3f}")
    print(f"  padded (repeat) : {rep_r:+.3f}   {'OK penalised' if rep_r < base_r else 'FAIL'}")
    print(f"  padded (filler) : {fil_r:+.3f}   {'OK penalised' if fil_r < base_r else 'FAIL'}")

    verdict = ok_peak and rep_r < base_r and fil_r < base_r
    print(f"\n[VERDICT] {'PASS' if verdict else 'FAIL'} "
          f"(peak-correct={ok_peak}, repeat-pad-penalised={rep_r < base_r}, "
          f"filler-pad-penalised={fil_r < base_r})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--model", default="deepseek-ai/Janus-Pro-1B")
    ap.add_argument("--data_json", default="corl/eval/test_split_levels.json")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n_hack", type=int, default=100)
    ap.add_argument("--w_rep", type=float, default=1.0)
    args = ap.parse_args()
    if args.validate:
        _validate(args)
    else:
        # quick smoke test of the shape of the knob
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            print(f"alpha={a:.2f} -> target_len={target_len(a):6.1f} tok")
