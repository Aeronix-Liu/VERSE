#!/usr/bin/env python3
"""Train-on-test — STRICT: a leak requires (question, answer, image-scene) all match.

False-positive sources ruled out:
  - qid numeric collisions (independent namespaces) -> ignored
  - Q-text template collisions (same action sequence, different maze image;
    same generic phrasing, different scene) -> require answer + scene match too

Leak criteria (ALL must hold):
  1. normalized question text exact match
  2. normalized answer match
  3. scene/image identity: same scene_id prefix in image path, OR same non-generic
     image basename. (For maze: same example<N>.png basename.)
"""
import json
import re
import random
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
SAMPLE_N = 1000
SEED = 42
GENERIC_BASENAME = re.compile(r"^\d{1,4}\.jpe?g$", re.I)
SCENE_RE = re.compile(r"(scene\d{4}_\d{2}|sample_\d+|example\d+|arkitscenes/[0-9a-f]+|d[0-9a-f]{8})", re.I)


def norm_q(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower()) if s else ""


def norm_ans(s):
    return re.sub(r"\s+", " ", str(s or "").strip().lower()) if s is not None else ""


def scene_keys(paths):
    """Extract scene-identity tokens from image paths."""
    out = set()
    if not paths:
        return out
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        ps = str(p)
        for m in SCENE_RE.finditer(ps):
            out.add(m.group(1).lower())
        bn = Path(ps).name
        if not GENERIC_BASENAME.match(bn):
            out.add(bn.lower())
    return out


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    train_rows = load_jsonl(TRAIN)
    print(f"Training data: {len(train_rows)} rows")

    rng = random.Random(SEED)
    sample = rng.sample(train_rows, min(SAMPLE_N, len(train_rows)))
    print(f"Sampled {len(sample)} training rows (seed={SEED})\n")

    # Build benchmark index keyed by normalized question -> list of (ans, scene_keys)
    bench_idx = {}
    for name, path in BENCHES.items():
        data = json.load(open(path))
        idx = defaultdict(list)
        for r in data:
            idx[norm_q(r.get("question", ""))].append(
                (norm_ans(r.get("answer")), scene_keys(r.get("image_path")), r)
            )
        bench_idx[name] = (len(data), idx)
        print(f"Benchmark {name}: {len(data)} rows")

    print(f"\n{'='*78}")
    print(f"STRICT leak check (Q + answer + scene all match), {len(sample)} sampled rows")
    print(f"{'='*78}")
    strict_leak = defaultdict(int)
    q_only = defaultdict(int)
    q_ans = defaultdict(int)
    strict_examples = defaultdict(list)
    total_strict = set()
    for r in sample:
        rq = norm_q(r.get("question", ""))
        ra = norm_ans(r.get("answer"))
        rsc = scene_keys(r.get("image_path"))
        if not rq:
            continue
        for name in BENCHES:
            _, idx = bench_idx[name]
            if rq not in idx:
                continue
            q_only[name] += 1
            for ba, bsc, bm in idx[rq]:
                if ra == ba:
                    q_ans[name] += 1
                    if rsc and bsc and (rsc & bsc):
                        strict_leak[name] += 1
                        total_strict.add(id(r))
                        if len(strict_examples[name]) < 5:
                            strict_examples[name].append((r, bm))
                        break
            break  # one bench per row for q_only counting simplicity

    print(f"\n{'benchmark':<28} {'Q-only':>7} {'Q+ans':>7} {'STRICT':>7} {'pct':>6}")
    for name in BENCHES:
        print(f"{name:<28} {q_only[name]:>7} {q_ans[name]:>7} {strict_leak[name]:>7} {strict_leak[name]/len(sample)*100:>5.1f}%")
    print(f"\nTotal STRICT leaks (Q+ans+scene): {len(total_strict)}/{len(sample)} = {len(total_strict)/len(sample)*100:.1f}%")

    print(f"\n{'='*78}")
    print("STRICT leak examples:")
    for name, exs in strict_examples.items():
        print(f"\n--- {name} ---")
        for tr, bm in exs[:3]:
            print(f"  train: data_type={tr.get('data_type')} qid={tr.get('question_id')} ans={tr.get('answer')}")
            print(f"  bench: qid={bm.get('question_id')} ans={bm.get('answer')}")
            print(f"  train img: {tr.get('image_path')}")
            print(f"  bench img: {bm.get('image_path')}")
            print(f"  Q: {tr.get('question','')[:160]}")
            print()

    print(f"{'='*78}")
    print("Extrapolation to full training set (38,610 rows):")
    for name in BENCHES:
        est = strict_leak[name] / len(sample) * len(train_rows)
        print(f"  {name}: ~{est:.0f} strict leaks ({strict_leak[name]/len(sample)*100:.1f}% of sample)")


if __name__ == "__main__":
    main()
