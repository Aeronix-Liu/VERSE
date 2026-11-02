#!/usr/bin/env python3
"""Train-on-test analysis for ViLaSR cold-start (SFT) and RRS stages.

Same methodology as the GRPO check:
  - Q-text exact match (normalized)
  - Q + answer match
  - STRICT: Q + answer + scene/image-identity match
For maze, also report Q+ans+grid_size matches and md5-verify images where possible.
"""
import json
import re
import hashlib
from collections import defaultdict
from pathlib import Path

BENCH_DIR = Path("/data/pengtianshuo/self_evolve/Xinran/ViLaSR/benchmark")
DATA_ROOT = Path("/data/pengtianshuo/self_evolve/Xinran/ViLaSR/ViLaSR-data")
BENCHES = {
    "maze": BENCH_DIR / "maze.json",
    "vsi_bench": BENCH_DIR / "vsi_bench.json",
    "spar_bench": BENCH_DIR / "spar_bench.json",
    "mmsi_bench": BENCH_DIR / "mmsi_bench.json",
    "SpatialEval_spatialreal": BENCH_DIR / "SpatialEval_spatialreal.json",
}
STAGES = {
    "cold_start": {
        "vqa": DATA_ROOT / "cold_start/vqa_clean.json",
        "maze": DATA_ROOT / "cold_start/maze_clean.json",
        "GPT4Scene": DATA_ROOT / "cold_start/GPT4Scene_clean.json",
        "SR_91k": DATA_ROOT / "cold_start/SR_91k_clean.json",
    },
    "rrs": {
        "vqa": DATA_ROOT / "reflective_rejection_sampling/vqa_clean.json",
        "maze": DATA_ROOT / "reflective_rejection_sampling/maze_clean.json",
        "GPT4Scene": DATA_ROOT / "reflective_rejection_sampling/GPT4Scene_clean.json",
        "SR_91k": DATA_ROOT / "reflective_rejection_sampling/SR_91k_clean.json",
    },
}
GENERIC = re.compile(r"^\d{1,4}\.jpe?g$", re.I)
SCENE_RE = re.compile(r"(scene\d{4}_\d{2}|sample_\d+|example\d+|arkitscenes/[0-9a-f]+|d[0-9a-f]{8})", re.I)


def norm_q(s): return re.sub(r"\s+", " ", (s or "").strip().lower()) if s else ""
def norm_ans(s): return re.sub(r"\s+", " ", str(s or "").strip().lower()) if s is not None else ""


def get_imgs(r):
    for k in ("image_path", "images"):
        v = r.get(k)
        if v: return v
    return []


def scene_keys(paths):
    out = set()
    if not paths: return out
    if isinstance(paths, str): paths = [paths]
    for p in paths:
        ps = str(p)
        for m in SCENE_RE.finditer(ps): out.add(m.group(1).lower())
        bn = Path(ps).name
        if not GENERIC.match(bn): out.add(bn.lower())
    return out


def md5(p):
    try: return hashlib.md5(open(p, "rb").read()).hexdigest()
    except Exception: return None


def load_json(path):
    return json.load(open(path))


def main():
    # Build benchmark indexes
    bench_idx = {}
    print("Benchmarks:")
    for name, path in BENCHES.items():
        data = load_json(path)
        idx = defaultdict(list)
        for r in data:
            idx[norm_q(r.get("question", ""))].append((norm_ans(r.get("answer")), scene_keys(get_imgs(r)), r))
        bench_idx[name] = (len(data), idx)
        print(f"  {name}: {len(data)} rows, {len(idx)} unique Q")

    # Process each stage
    for stage, files in STAGES.items():
        print(f"\n{'='*78}")
        print(f"STAGE: {stage}")
        print(f"{'='*78}")
        all_rows = []
        per_source = defaultdict(int)
        for src, path in files.items():
            data = load_json(path)
            print(f"  {src}: {len(data)} rows ({path.name})")
            for r in data:
                r["_src_file"] = src
                all_rows.append(r)
                per_source[r.get("source", src)] += 1
        print(f"  TOTAL: {len(all_rows)} rows  by source: {dict(per_source)}")

        # Leak check
        q_only = defaultdict(int)
        q_ans = defaultdict(int)
        strict = defaultdict(int)
        maze_qans_grid = defaultdict(int)
        total_strict = 0
        for r in all_rows:
            rq = norm_q(r.get("question", "")); ra = norm_ans(r.get("answer")); rsc = scene_keys(get_imgs(r))
            if not rq: continue
            for name in BENCHES:
                _, idx = bench_idx[name]
                if rq not in idx: continue
                q_only[name] += 1
                for ba, bsc, bm in idx[rq]:
                    if ra == ba:
                        q_ans[name] += 1
                        if rsc and bsc and (rsc & bsc):
                            strict[name] += 1
                            total_strict += 1
                            break
                # maze grid check
                if name == "maze" and rq in idx:
                    for ba, bsc, bm in idx[rq]:
                        if ra == ba:
                            tg = r.get("grid_size"); bg = bm.get("grid_size")
                            if tg and bg and tg == bg:
                                maze_qans_grid[name] += 1
                            break
                break  # one bench for q_only

        print(f"\n  Leak counts (full {stage} set, {len(all_rows)} rows):")
        print(f"  {'benchmark':<28} {'Q-only':>8} {'Q+ans':>8} {'STRICT':>8}")
        for name in BENCHES:
            print(f"  {name:<28} {q_only[name]:>8} {q_ans[name]:>8} {strict[name]:>8}")
        print(f"\n  maze Q+ans+grid_size matches: {maze_qans_grid['maze']}")
        print(f"  TOTAL strict leaks: {total_strict}/{len(all_rows)} = {total_strict/max(len(all_rows),1)*100:.2f}%")

        # md5 verify maze Q+ans matches (sample up to 30)
        if q_ans["maze"] > 0:
            checked = same = diff = 0
            for r in all_rows:
                if checked >= 30: break
                if r.get("source") != "maze" and r.get("_src_file") != "maze": continue
                rq = norm_q(r.get("question", "")); ra = norm_ans(r.get("answer"))
                _, idx = bench_idx["maze"]
                if rq in idx:
                    for ba, bsc, bm in idx[rq]:
                        if ra == ba:
                            # resolve sft image (first frame) vs bench image
                            sft_imgs = get_imgs(r)
                            tp = DATA_ROOT / sft_imgs[0].replace("./", "", 1) if sft_imgs else None
                            bp = BENCH_DIR / get_imgs(bm)[0] if get_imgs(bm) else None
                            tm, bmh = md5(tp), md5(bp)
                            if tm and bmh:
                                if tm == bmh: same += 1
                                else: diff += 1
                                checked += 1
                            break
            print(f"  maze md5-verify: checked {checked} Q+ans pairs -> identical image={same}, different image={diff}")

    print(f"\n{'='*78}")
    print("NOTE: SFT maze images are stored as multiple frames (image_0.png..) under")
    print("cold_start/maze/<qid>/images/, while benchmark maze uses maze/grid<N>/example<M>.png.")
    print("Different image formats -> md5 will differ even for the same maze; Q+ans+grid is the")
    print("best available signal for maze, and it shows coincidences (same action seq + dest on")
    print("different layouts), not true leaks.")


if __name__ == "__main__":
    main()
