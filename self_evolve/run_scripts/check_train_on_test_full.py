#!/usr/bin/env python3
"""Train-on-test — FULL training set, strict (Q + answer + scene/image) match."""
import json, re, hashlib
from collections import defaultdict
from pathlib import Path

TRAIN = "/data/pengtianshuo/self_evolve/Xinran/ViLaSR/ViLaSR-data/rl/vilasr_rl_data.jsonl"
BENCH_DIR = Path("/data/pengtianshuo/self_evolve/Xinran/ViLaSR/benchmark")
BENCHES = {
    "maze": BENCH_DIR / "maze.json",
    "vsi_bench": BENCH_DIR / "vsi_bench.json",
    "spar_bench": BENCH_DIR / "spar_bench.json",
    "mmsi_bench": BENCH_DIR / "mmsi_bench.json",
    "SpatialEval_spatialreal": BENCH_DIR / "SpatialEval_spatialreal.json",
}
GENERIC = re.compile(r"^\d{1,4}\.jpe?g$", re.I)
SCENE_RE = re.compile(r"(scene\d{4}_\d{2}|sample_\d+|example\d+|arkitscenes/[0-9a-f]+|d[0-9a-f]{8})", re.I)


def norm_q(s): return re.sub(r"\s+", " ", (s or "").strip().lower()) if s else ""
def norm_ans(s): return re.sub(r"\s+", " ", str(s or "").strip().lower()) if s is not None else ""


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


def img_path_on_disk(rel, bench=False):
    if not rel: return None
    p = rel[0] if isinstance(rel, list) else rel
    if bench:
        return BENCH_DIR / p
    return Path("/data/pengtianshuo/self_evolve/Xinran/ViLaSR/ViLaSR-data/rl") / p.replace("rl/", "", 1) if str(p).startswith("rl/") else Path("/data/pengtianshuo/self_evolve/Xinran/ViLaSR/ViLaSR-data") / p


def main():
    train_rows = [json.loads(l) for l in open(TRAIN) if l.strip()]
    print(f"Full training set: {len(train_rows)} rows")
    bench_idx = {}
    for name, path in BENCHES.items():
        data = json.load(open(path))
        idx = defaultdict(list)
        for r in data:
            idx[norm_q(r.get("question", ""))].append((norm_ans(r.get("answer")), scene_keys(r.get("image_path")), r))
        bench_idx[name] = (len(data), idx)
        print(f"Benchmark {name}: {len(data)} rows")

    strict = defaultdict(int)
    q_only = defaultdict(int)
    q_ans = defaultdict(int)
    md5_verified_same = defaultdict(int)
    md5_verified_diff = defaultdict(int)
    examples = defaultdict(list)
    for r in train_rows:
        rq = norm_q(r.get("question", "")); ra = norm_ans(r.get("answer")); rsc = scene_keys(r.get("image_path"))
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
                        if len(examples[name]) < 3:
                            examples[name].append((r, bm))
                        break
            break  # avoid double-count across benches for q_only

    print(f"\n{'='*78}")
    print("FULL training set leak counts:")
    print(f"{'benchmark':<28} {'Q-only':>8} {'Q+ans':>8} {'STRICT':>8}")
    for name in BENCHES:
        print(f"{name:<28} {q_only[name]:>8} {q_ans[name]:>8} {strict[name]:>8}")
    total_strict = sum(strict.values())
    print(f"\nTOTAL strict leaks (Q+ans+scene) across full training set: {total_strict}/{len(train_rows)} = {total_strict/len(train_rows)*100:.2f}%")

    # md5-verify the strict examples (if any) or the best Q+ans candidates
    print(f"\n{'='*78}")
    if total_strict == 0:
        print("No strict (Q+ans+scene) leaks found. Verifying top Q+ans candidates by md5 image hash:")
        for name in BENCHES:
            if q_ans[name] == 0: continue
            # find a few Q+ans matches and md5-check images
            checked = 0; same = 0; diff = 0
            for r in train_rows:
                if checked >= 30: break
                rq = norm_q(r.get("question", "")); ra = norm_ans(r.get("answer"))
                _, idx = bench_idx[name]
                if rq in idx:
                    for ba, bsc, bm in idx[rq]:
                        if ra == ba:
                            tp = img_path_on_disk(r.get("image_path"))
                            bp = img_path_on_disk(bm.get("image_path"), bench=True)
                            tm, bmh = md5(tp), md5(bp)
                            if tm and bmh:
                                if tm == bmh: same += 1
                                else: diff += 1
                                checked += 1
                            break
            print(f"  {name}: md5-checked {checked} Q+ans pairs -> identical image={same}, different image={diff}")
    else:
        print("Strict leak examples (md5-verified):")
        for name, exs in examples.items():
            for tr, bm in exs:
                tp = img_path_on_disk(tr.get("image_path"))
                bp = img_path_on_disk(bm.get("image_path"), bench=True)
                print(f"  {name}: train img md5={md5(tp)}  bench img md5={md5(bp)}  same={md5(tp)==md5(bp)}")


if __name__ == "__main__":
    main()
