"""Build a per-image attribute sidecar: {id, modality, pose, is_grid, src}.

Mirrors build_caption_cache.py's infrastructure (DDP sharding, per-rank
resumable JSONL shards, merge step) but for *labeling* instead of captioning.

Attributes
----------
modality : CT|MRI|PET|X-ray|Ultrasound|Histopathology|Fundus|Endoscopy|
           Dermoscopy|Mammography|Angiography|Chart|Other|unknown
pose     : axial|coronal|sagittal|AP|PA|lateral|oblique|none|unknown
is_grid  : single|multi|unknown   (binary layout; "filter only" plan)

Phases (run in order; the .sh launcher does text -> grid -> vlm -> merge):
  text  : regex over the K4 cached captions. CPU-only, single process. Writes
          --text_out and prints coverage. Run first.
  grid  : projection-profile layout detection for rows the captions left without
          is_grid. CPU multiprocessing; rewrites --text_out in place. This is the
          default is_grid source, so the VLM only handles modality/pose gaps.
  vlm   : DDP-sharded Janus i2t over rows where modality/pose is still unknown.
          Resumable per-rank shards in --out_dir. Launch with torchrun.
  merge : overlay shard results onto the text rows -> --merged_out.

Caption-first + projection-profile layout keeps most images off the VLM.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

# ----------------------------- keyword tables -----------------------------
# Ordered most-specific -> least so the first hit wins (e.g. "PET/CT" -> PET).
MODALITY_PATTERNS = [
    ("Histopathology", r"histolog|histopatholog|h&e|hematoxylin|haematoxylin|eosin|immunohistochem|\bihc\b|microscop|cytolog|biopsy|\bstain(ed|ing)?\b|patholog"),
    ("Fundus",         r"\bfundus\b|fundoscop|ophthalmoscop|retinal photograph|\boct\b|optical coherence"),
    ("Dermoscopy",     r"dermoscop|dermatoscop|skin lesion|cutaneous"),
    ("Endoscopy",      r"endoscop|colonoscop|gastroscop|laparoscop|bronchoscop|cystoscop|arthroscop"),
    ("Mammography",    r"mammogra"),
    ("Angiography",    r"angiogra"),
    ("Ultrasound",     r"ultrasound|ultrasonograph|sonograph|\bus\b image|echocardiogra|doppler"),
    ("PET",            r"positron emission|\bpet\b|pet[\/-]ct|pet[\/-]mri"),
    ("MRI",            r"magnetic resonance|\bmri\b|\bmr\b image|t1[- ]weighted|t2[- ]weighted|\bflair\b|diffusion[- ]weighted"),
    ("CT",             r"computed tomograph|\bct\b scan|\bct\b image|\bcect\b|\bhrct\b|axial ct"),
    ("X-ray",          r"x[- ]?ray|radiograph|chest film|\bcxr\b|plain film|roentgen"),
    ("Chart",          r"\bgraph\b|\bchart\b|\bplot\b|heatmap|bar chart|scatter|flowchart|schematic|diagram|\btable\b|boxplot|histogram"),
]

POSE_PATTERNS = [
    ("axial",    r"\baxial\b|transverse|transaxial|cross[- ]section"),
    ("coronal",  r"\bcoronal\b"),
    ("sagittal", r"\bsagittal\b"),
    ("PA",       r"posteroanterior|\bpa\b view|pa projection"),
    ("AP",       r"anteroposterior|\bap\b view|ap projection"),
    ("lateral",  r"\blateral\b view|lateral projection|\blateral\b radiograph"),
    ("oblique",  r"\boblique\b"),
]

# Presence of any => multi-panel. Absence != single, so -> unknown.
GRID_PATTERN = re.compile(
    r"\bpanel(s)?\b|top row|bottom row|upper row|lower row|left panel|right panel|"
    r"montage|composite image|subfigure|columns? labeled|rows? labeled|"
    r"panels? labeled|\(\s*[a-f]\s*\)|labeled [a-f] (through|to) [a-f]",
    re.IGNORECASE,
)

MODALITY_RE = [(name, re.compile(p, re.IGNORECASE)) for name, p in MODALITY_PATTERNS]
POSE_RE = [(name, re.compile(p, re.IGNORECASE)) for name, p in POSE_PATTERNS]

# Pose only meaningfully applies to these modalities; elsewhere pose="none"
# (a rule-based fill, applied in run_merge). Histology slides, fundus,
# dermoscopy, etc. have no axial/coronal/sagittal so unknown != missing.
POSE_MODS = {"CT", "MRI", "PET", "X-ray", "Angiography", "Mammography"}

# VLM pose synonyms that should map to "none" when the strict canonical lookup
# misses (model says "n/a" instead of "none", etc.).
_POSE_NONE_SYNONYMS = {
    "none", "n/a", "na", "unknown", "not applicable", "not specified",
    "none specified", "any", "various", "multiple", "-", "--",
}


def _fix_janus_text(s: str) -> str:
    return s.replace("Ġ", " ").replace("Ċ", "\n").strip()


def caption_text(rec):
    c = rec.get("cached_captions")
    if isinstance(c, (list, tuple)):
        return " ".join(str(x) for x in c)
    return str(c) if c is not None else ""


def extract_text_attrs(text):
    modality = next((name for name, rx in MODALITY_RE if rx.search(text)), None)
    pose = next((name for name, rx in POSE_RE if rx.search(text)), None)
    is_grid = "multi" if GRID_PATTERN.search(text) else None
    return modality, pose, is_grid


def needs_vlm(row):
    return row["modality"] is None or row["pose"] is None or row["is_grid"] is None


# ----------------------- projection-profile grid detector -----------------------
# A montage's panels are separated by near-uniform gutter bands (solid white or
# black). Collapsing the image to per-row / per-column variance, a gutter shows
# up as a low-variance, extreme-mean line. An *interior* gutter band => multi.
GRID_DOWNSCALE = 256      # longest side after resize (speed + denoise)
GRID_VAR_FRAC = 0.01      # line counts as flat if its var < frac * global var
GRID_FLAT_HI = 240.0      # ...and its mean is near white
GRID_FLAT_LO = 15.0       # ...or near black
GRID_BORDER_FRAC = 0.06   # ignore gutters within this fraction of each edge
GRID_MIN_BAND = 2         # min contiguous flat lines (downscaled px) for a band


def _count_interior_bands(line_var, line_mean, n, global_var):
    import numpy as np
    thr = GRID_VAR_FRAC * global_var
    flat = (line_var < thr) & ((line_mean > GRID_FLAT_HI) | (line_mean < GRID_FLAT_LO))
    b = int(GRID_BORDER_FRAC * n)
    if b:
        flat[:b] = False
        flat[n - b:] = False
    bands = run = 0
    for v in flat:
        if v:
            run += 1
        else:
            bands += run >= GRID_MIN_BAND
            run = 0
    bands += run >= GRID_MIN_BAND
    return bands


def detect_grid_projection(path):
    """Return 'single' or 'multi' from row/column variance profiles. Robust to
    clean gutters; borderless/colored-separator montages are the known misses."""
    import numpy as np
    from PIL import Image
    try:
        im = Image.open(path).convert("L")
    except Exception:
        return None
    w, h = im.size
    scale = GRID_DOWNSCALE / max(w, h)
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    gv = float(a.var()) + 1e-6
    v_bands = _count_interior_bands(a.var(axis=0), a.mean(axis=0), a.shape[1], gv)
    h_bands = _count_interior_bands(a.var(axis=1), a.mean(axis=1), a.shape[0], gv)
    return "multi" if (v_bands >= 1 or h_bands >= 1) else "single"


def _grid_worker(path):
    return detect_grid_projection(path)


# ----------------------------- phase: text -----------------------------
def run_text(args):
    data = json.load(open(args.captions))
    if args.max_samples:
        data = data[: args.max_samples]
    out_path = Path(args.text_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mod_hist, pose_hist = Counter(), Counter()
    grid_multi = need_vlm = 0
    with out_path.open("w") as f:
        for rec in data:
            modality, pose, is_grid = extract_text_attrs(caption_text(rec))
            mod_hist[modality or "unknown"] += 1
            pose_hist[pose or "unknown"] += 1
            grid_multi += int(is_grid == "multi")
            row = {
                "id": rec.get("id"),
                "image": rec.get("image"),
                "modality": modality,
                "pose": pose,
                "is_grid": is_grid,
                "src": {
                    "modality": "caption" if modality else None,
                    "pose": "caption" if pose else None,
                    "is_grid": "caption" if is_grid else None,
                },
            }
            need_vlm += int(needs_vlm(row))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(data)
    def pct(c): return f"{c} ({100 * c / n:.1f}%)"
    print(f"\n=== text-phase coverage over {n} rows ===")
    print(f"wrote {out_path}\n")
    print(f"modality resolved: {pct(n - mod_hist['unknown'])}")
    for name, c in mod_hist.most_common():
        print(f"    {name:<16} {pct(c)}")
    print(f"pose resolved:     {pct(n - pose_hist['unknown'])}")
    for name, c in pose_hist.most_common():
        print(f"    {name:<16} {pct(c)}")
    print(f"is_grid==multi (caption-detected): {pct(grid_multi)}")
    print(f"\nrows with any attr unknown after text: {pct(need_vlm)}")
    print("NEXT: --phase grid fills is_grid via projection profiles (cheap, CPU), "
          "then --phase vlm only handles the residual modality/pose gaps.")


# ----------------------------- phase: vlm (DDP) -----------------------------
VLM_INSTRUCT = (
    "You are labeling a single medical figure. Answer ONLY in this exact format, "
    "one per line, no extra text:\n"
    "Modality: <CT|MRI|PET|X-ray|Ultrasound|Histopathology|Fundus|Endoscopy|"
    "Dermoscopy|Mammography|Angiography|Chart|Other>\n"
    "View: <axial|coronal|sagittal|AP|PA|lateral|oblique|none>\n"
    "Layout: <single|grid>"
)

_MOD_CANON = {m.lower(): m for m, _ in MODALITY_PATTERNS}
_MOD_CANON["other"] = "Other"
_POSE_CANON = {p.lower(): p for p, _ in POSE_PATTERNS}
_POSE_CANON["none"] = "none"


def parse_vlm(textout):
    t = _fix_janus_text(textout).lower()
    mod = next((_MOD_CANON[k] for k in _MOD_CANON if re.search(rf"modality:\s*{re.escape(k)}", t)), None)
    pose = next((_POSE_CANON[k] for k in _POSE_CANON if re.search(rf"view:\s*{re.escape(k)}", t)), None)
    # Fallback: accept common "no pose applicable" synonyms as "none".
    if pose is None:
        m = re.search(r"view:\s*([a-z/\.\-\s]+)", t)
        if m:
            val = m.group(1).strip().rstrip(".,;").splitlines()[0].strip()
            if val in _POSE_NONE_SYNONYMS:
                pose = "none"
    grid = None
    m = re.search(r"layout:\s*(single|grid|multi)", t)
    if m:
        grid = "single" if m.group(1) == "single" else "multi"
    return mod, pose, grid


def _image_path(data_dir, img):
    rel = img[0] if isinstance(img, (list, tuple)) else img
    return os.path.join(data_dir, rel)


def run_vlm(args):
    import torch
    import torch.distributed as dist
    from PIL import Image
    from transformers import AutoModelForCausalLM
    from janus.models import VLChatProcessor

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    text_path = Path(args.text_out)
    if not text_path.exists():
        sys.exit(f"{text_path} not found -- run --phase text first")
    rows = [json.loads(l) for l in text_path.open()]
    todo_all = [r for r in rows if needs_vlm(r)]

    # Contiguous shard for this rank.
    n = len(todo_all)
    per_rank = (n + world_size - 1) // world_size
    my_rows = todo_all[rank * per_rank: min((rank + 1) * per_rank, n)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"attr_shard_rank{rank:02d}.jsonl"
    # Smart resume: only skip rows whose shard line is FULLY resolved. If a prior
    # line still has a None attr (parser failed last time), re-query it. Combined
    # with attr-level overlay in run_merge, this safely chases residual gaps
    # without rerunning everything.
    done = set()
    if shard_path.exists():
        for l in shard_path.open():
            rec = json.loads(l)
            if (rec.get("modality") is not None
                    and rec.get("pose") is not None
                    and rec.get("is_grid") is not None):
                done.add(rec["id"])
    todo = [r for r in my_rows if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[rank {rank}] {len(my_rows)} rows ({len(done)} done) -> {len(todo)} to do", flush=True)

    processor = VLChatProcessor.from_pretrained(args.model)
    processor.system_prompt = ""
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    eos_id = processor.tokenizer.eos_token_id
    bos_id = processor.tokenizer.bos_token_id

    t0 = time.perf_counter()
    n_sess = 0
    with shard_path.open("a") as fout:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start: start + args.batch_size]
            convs, imgs, kept = [], [], []
            for r in batch:
                p = _image_path(args.data_dir, r["image"])
                if not os.path.exists(p):
                    continue
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    print(f"[rank {rank}] skip {p}: {e}", flush=True)
                    continue
                convs.append([
                    {"role": "<|User|>", "content": f"<image_placeholder>\n{VLM_INSTRUCT}"},
                    {"role": "<|Assistant|>", "content": ""},
                ])
                kept.append(r)
            if not kept:
                continue
            with torch.inference_mode():
                prep = processor(conversations=convs, images=[[im] for im in imgs],
                                 force_batchify=True).to(device)
                embeds = model.prepare_inputs_embeds(**prep)
                out_ids = model.language_model.generate(
                    inputs_embeds=embeds,
                    attention_mask=prep.attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=eos_id, bos_token_id=bos_id, eos_token_id=eos_id,
                )
            decoded = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            for r, txt in zip(kept, decoded):
                v_mod, v_pose, v_grid = parse_vlm(txt)
                if r["modality"] is None and v_mod:
                    r["modality"], r["src"]["modality"] = v_mod, "vlm"
                if r["pose"] is None and v_pose:
                    r["pose"], r["src"]["pose"] = v_pose, "vlm"
                if r["is_grid"] is None and v_grid:
                    r["is_grid"], r["src"]["is_grid"] = v_grid, "vlm"
                # Keep the raw VLM string so a future loosened parser can
                # re-extract attributes without re-querying Janus.
                r["_vlm_raw"] = _fix_janus_text(txt)[:240]
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            n_sess += len(kept)
            if (start // args.batch_size) % 5 == 0:
                rate = n_sess / max(time.perf_counter() - t0, 1e-3)
                eta = (len(todo) - n_sess) / max(rate, 1e-3) / 3600
                print(f"[rank {rank}] {n_sess}/{len(todo)} ({rate:.1f} img/s, ETA {eta:.2f}h)", flush=True)
    print(f"[rank {rank}] done -> {shard_path}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


# ----------------------------- phase: grid -----------------------------
def run_grid(args):
    """Fill is_grid for rows the captions left unknown, via projection profiles.
    Rewrites --text_out in place (atomic). Idempotent: only touches None rows."""
    from concurrent.futures import ProcessPoolExecutor

    text_path = Path(args.text_out)
    if not text_path.exists():
        sys.exit(f"{text_path} not found -- run --phase text first")
    rows = [json.loads(l) for l in text_path.open()]
    todo = [r for r in rows if r["is_grid"] is None and os.path.exists(_image_path(args.data_dir, r["image"]))]
    print(f"grid: {len(todo)} rows to detect (of {len(rows)} total)")

    paths = [_image_path(args.data_dir, r["image"]) for r in todo]
    workers = args.num_workers or os.cpu_count()
    counts = Counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r, label in zip(todo, ex.map(_grid_worker, paths, chunksize=64)):
            if label:
                r["is_grid"], r["src"]["is_grid"] = label, "projection"
                counts[label] += 1

    tmp = text_path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, text_path)
    n = len(rows)
    still = sum(needs_vlm(r) for r in rows)
    print(f"grid filled: {dict(counts)} -> rewrote {text_path}")
    print(f"rows still needing VLM (modality/pose gaps): {still} ({100*still/n:.1f}%)")


# ----------------------------- phase: merge -----------------------------
def run_merge(args):
    rows = {json.loads(l)["id"]: json.loads(l) for l in Path(args.text_out).open()}
    shards = sorted(Path(args.out_dir).glob("attr_shard_rank*.jsonl"))
    n_overlay = 0
    for s in shards:
        for line in s.open():
            r = json.loads(line)
            cur = rows.get(r["id"])
            if cur is None:
                rows[r["id"]] = r
                n_overlay += 1
                continue
            # Attr-level overlay: only non-None VLM values fill unset attrs.
            # A later failed-parse line can never clobber an earlier success,
            # so re-running the VLM on residual gaps is always safe.
            for k in ("modality", "pose", "is_grid"):
                v = r.get(k)
                if v is not None and cur.get(k) is None:
                    cur[k] = v
                    cur.setdefault("src", {})[k] = r.get("src", {}).get(k, "vlm")
            if r.get("_vlm_raw"):
                cur["_vlm_raw"] = r["_vlm_raw"]
            n_overlay += 1
    # Rule: pose is not applicable for non-pose modalities -> "none".
    n_rule = 0
    for r in rows.values():
        if (r.get("pose") is None
                and r.get("modality") is not None
                and r["modality"] not in POSE_MODS):
            r["pose"] = "none"
            r.setdefault("src", {})["pose"] = "rule:non-pose-modality"
            n_rule += 1
    out = Path(args.merged_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(list(rows.values()), f, ensure_ascii=False)
    still = {k: sum(r.get(k) is None for r in rows.values()) for k in ("modality", "pose", "is_grid")}
    print(f"merged {len(rows)} rows ({n_overlay} lines from {len(shards)} VLM shards) -> {out}")
    print(f"rule fill (pose='none' for non-pose modalities): {n_rule}")
    print(f"rows still missing per attr: {still}")


def main():
    host = os.uname().nodename.split(".")[0]
    default_dd = {
        "cvssp-retina03": "/work/um00109/MLLM/datasets/PubMedVision",
        "ulws072": "/vol/research/fmodel_medical/people/umar/datasets/PubMedVision",
    }.get(host, "/vol/research/fmodel_medical/people/umar/datasets/PubMedVision")

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="text", choices=["text", "grid", "vlm", "merge"])
    ap.add_argument("--captions", default=os.path.join(default_dd, "PubMedVision_CachedCaptions_K4.json"))
    ap.add_argument("--data_dir", default=default_dd)
    ap.add_argument("--text_out", default="data/attribute_sidecar.text.jsonl")
    ap.add_argument("--out_dir", default="data/attribute_sidecar_shards")
    ap.add_argument("--merged_out", default="data/attribute_sidecar.json")
    ap.add_argument("--model", default="deepseek-ai/Janus-Pro-1B")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--max_samples", type=int, default=0, help="cap text rows (debug)")
    ap.add_argument("--limit", type=int, default=0, help="cap per-rank VLM rows (trial)")
    ap.add_argument("--num_workers", type=int, default=0, help="grid procs (0=os.cpu_count())")
    args = ap.parse_args()

    if args.phase == "text":
        run_text(args)
    elif args.phase == "grid":
        run_grid(args)
    elif args.phase == "vlm":
        run_vlm(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
