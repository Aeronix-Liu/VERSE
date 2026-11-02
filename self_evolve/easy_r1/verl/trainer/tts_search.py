"""Verifier-Guided rollout Search (VGS) selection for Round-2 self-evolve GRPO.

Xinran (2026-08-18). Train-time search on top of the existing two-round
self-evolve GRPO loop, WITHOUT touching the delicate agentic rollout token path:

  * For every trained step (maze / VSI), the Round-2 solver frontier is widened
    from `rollout.n` to `effective_n = rollout.n * B` i.i.d. agentic rollouts per
    context (done via a per-call `meta_info["n"]` override in the trainer).
  * Each rollout is scored by the DETERMINISTIC engine — answer correctness
    (DGE / maze letter) plus, for maze, the Path-Tracer trajectory fidelity
    (LCS of drawn cell path vs the BFS ground-truth cell path). This is a
    zero-LLM, un-gameable verifier.
  * We then SELECT `keep = rollout.n` rollouts per group to seed the GRPO update.

Selection policy — "guarantee positive, preserve contrast":
  The value of a wider search is that it finds BETTER engine-verified positives
  than i.i.d. sampling would. But naive top-n throws away the negative tail that
  GRPO needs (its negative advantage is what suppresses error modes; see
  cursor_memory 2026-08-18). So when a group has both correct and wrong rollouts
  we keep the best positives AND a floor of negatives; when a group is saturated
  (all correct) we rank by fidelity so faithful-vs-Potemkin still gives a
  gradient (exactly the faithfulness signal we want on maze). A variance
  fallback reverts a group to a random keep whenever the selected rewards would
  be (near-)identical, so advantage never collapses to zero below baseline.

The core selection (`select_indices`) is a pure function over per-sample
(acc, fidelity, is_valid, task_type) arrays and is unit-tested on CPU.
"""

import json
import os
import sys
from collections import defaultdict
from random import Random
from typing import Any, Dict, List, Optional, Tuple

# The engine scorers live in the reward-function package (CPU-only, no torch).
_REWARD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "reward_function")
)
if _REWARD_DIR not in sys.path:
    sys.path.insert(0, _REWARD_DIR)

from agentic_solver_reward import (  # noqa: E402
    _accuracy_for_valid,
    maze_trajectory_fidelity,
    parse_bbox_and_movement,
    parse_dialog,
)
from majority_correctness_reward import _extract_last_answer_tag  # noqa: E402

MAZE_TASK_TYPE = "maze_navigation"
_CORRECT_THRESHOLD = 0.5  # acc >= this counts as "correct" for class balancing


def task_bucket(task_type: str) -> str:
    """Map a fine-grained task type to a coarse TTS bucket (maze / vsi / spar)."""
    t = str(task_type or "")
    if t == MAZE_TASK_TYPE:
        return "maze"
    if t.startswith("spar_"):
        return "spar"
    return "vsi"


def score_rollout(
    *,
    task_type: str,
    sequence_text: str,
    response_text: str,
    gt: str,
    mc_options_json: str = "",
    gt_letter: str = "",
    answer_style: str = "",
    maze_ref_json: str = "",
    is_valid: bool = True,
) -> Tuple[float, float]:
    """Return (accuracy, maze_fidelity) for one solver rollout via the engine.

    Invalid questions (the "explain why invalid" branch) have no cheap engine
    accuracy, so they return (0, 0) and are handled as a keep-first-n pool.
    """
    if not is_valid:
        return 0.0, 0.0

    seq = str(sequence_text or "")
    resp = str(response_text or "")
    answer_text = _extract_last_answer_tag(resp) or _extract_last_answer_tag(seq) or ""

    acc = 0.0
    try:
        acc = float(
            _accuracy_for_valid(
                answer_text,
                gt,
                task_type,
                mc_options_json=str(mc_options_json or ""),
                gt_letter=str(gt_letter or ""),
                answer_style=str(answer_style or ""),
            )
        )
    except Exception:
        acc = 0.0

    fidelity = 0.0
    if str(task_type or "") == MAZE_TASK_TYPE and maze_ref_json:
        try:
            ref = json.loads(maze_ref_json)
            if isinstance(ref, dict) and ref.get("grid_size"):
                conversations = parse_dialog(seq) if seq else []
                assistant_text = "".join(
                    turn["content"] for turn in conversations if turn.get("role") == "assistant"
                )
                _, movement_list, _, _ = parse_bbox_and_movement(assistant_text)
                fid = maze_trajectory_fidelity(movement_list, ref)
                fidelity = float(fid.get("fidelity", 0.0))
        except Exception:
            fidelity = 0.0

    return acc, fidelity


def select_indices(
    *,
    uids: List[str],
    accs: List[float],
    fids: List[float],
    is_valids: List[bool],
    keep: int,
    fidelity_weight: float = 0.5,
    min_neg_frac: float = 0.25,
    seed: int = 0,
    var_eps: float = 1e-6,
) -> Tuple[List[int], Dict[str, float]]:
    """Pure selection over per-sample engine scores.

    Groups samples by uid (order preserved). For each group of size > keep,
    selects `keep` indices with the class-balanced + fidelity + variance-fallback
    policy. Groups already <= keep are kept whole. Returns (global kept indices,
    aggregate metrics).
    """
    rng = Random(seed)
    grouped: "defaultdict[str, List[int]]" = defaultdict(list)
    order: List[str] = []
    for i, uid in enumerate(uids):
        key = str(uid)
        if key not in grouped:
            order.append(key)
        grouped[key].append(i)

    kept: List[int] = []
    n_groups = 0
    n_searched = 0
    n_fallback = 0
    kept_pos = 0
    kept_total = 0
    mean_before = 0.0
    mean_after = 0.0

    for key in order:
        idxs = grouped[key]
        n_groups += 1
        if len(idxs) <= keep:
            kept.extend(idxs)
            for i in idxs:
                kept_total += 1
                if accs[i] >= _CORRECT_THRESHOLD:
                    kept_pos += 1
            continue

        n_searched += 1
        group_valid = bool(is_valids[idxs[0]])
        scores = {i: accs[i] + fidelity_weight * fids[i] for i in idxs}
        mean_before += sum(scores[i] for i in idxs) / len(idxs)

        if not group_valid:
            # No cheap engine accuracy for invalid groups: keep first `keep`
            # (they are only the "explain why invalid" branch).
            sel = list(idxs[:keep])
        else:
            correct = [i for i in idxs if accs[i] >= _CORRECT_THRESHOLD]
            wrong = [i for i in idxs if accs[i] < _CORRECT_THRESHOLD]
            if correct and wrong:
                min_neg = max(1, int(round(keep * min_neg_frac)))
                max_pos = max(1, keep - min_neg)
                pos_sorted = sorted(correct, key=lambda i: (scores[i], rng.random()), reverse=True)
                sel = pos_sorted[: min(max_pos, len(pos_sorted))]
                # Fill with a diversity-preserving random draw of negatives.
                need = keep - len(sel)
                neg_pool = list(wrong)
                rng.shuffle(neg_pool)
                sel += neg_pool[:need]
                if len(sel) < keep:  # not enough negatives -> top up with more positives
                    rest = [i for i in pos_sorted if i not in set(sel)]
                    sel += rest[: keep - len(sel)]
            else:
                # Saturated group (all correct or all wrong): rank by combined
                # score. For all-correct maze this keeps the most faithful ones.
                sel = sorted(idxs, key=lambda i: (scores[i], rng.random()), reverse=True)[:keep]

            # Variance fallback: if the selected rewards are ~identical, advantage
            # would vanish -> revert to a random keep (restores natural spread).
            sel_scores = [scores[i] for i in sel]
            if (max(sel_scores) - min(sel_scores)) < var_eps:
                pool = list(idxs)
                rng.shuffle(pool)
                sel = pool[:keep]
                n_fallback += 1

        kept.extend(sel)
        mean_after += sum(scores[i] for i in sel) / max(1, len(sel))
        for i in sel:
            kept_total += 1
            if accs[i] >= _CORRECT_THRESHOLD:
                kept_pos += 1

    metrics = {
        "groups": float(n_groups),
        "groups_searched": float(n_searched),
        "zero_var_fallback": float(n_fallback),
        "zero_var_fallback_rate": (float(n_fallback) / n_searched) if n_searched else 0.0,
        "kept_pos_ratio": (float(kept_pos) / kept_total) if kept_total else 0.0,
        "mean_score_before": (mean_before / n_searched) if n_searched else 0.0,
        "mean_score_after": (mean_after / n_searched) if n_searched else 0.0,
    }
    return kept, metrics


def _get_nt(new_batch, key: str, n: int) -> List[Any]:
    nt = new_batch.non_tensor_batch
    if key in nt and nt[key] is not None:
        return list(nt[key])
    return [None] * n


def select_for_batch(
    new_batch,
    *,
    group_size: int,
    keep: int,
    fidelity_weight: float = 0.5,
    seed: int = 0,
) -> Tuple[List[int], Dict[str, float]]:
    """Glue: extract fields from the Round-2 DataProto, score every rollout with
    the engine, and select `keep` per uid group. Returns (kept indices, metrics).
    """
    n = len(new_batch)
    uids = [str(u) for u in _get_nt(new_batch, "uid", n)]
    task_types = _get_nt(new_batch, "question_type", n)
    gts = _get_nt(new_batch, "gt", n)
    mc = _get_nt(new_batch, "mc_options_json", n)
    gtl = _get_nt(new_batch, "gt_letter", n)
    astyle = _get_nt(new_batch, "answer_style", n)
    maze_refs = _get_nt(new_batch, "maze_fidelity_ref_json", n)
    valids = _get_nt(new_batch, "is_valid", n)
    seq_texts = _get_nt(new_batch, "agentic_sequence_text", n)
    resp_texts = _get_nt(new_batch, "agentic_response_text", n)

    accs: List[float] = [0.0] * n
    fids: List[float] = [0.0] * n
    is_valids: List[bool] = [True] * n
    for i in range(n):
        iv = True if valids[i] is None else bool(valids[i])
        is_valids[i] = iv
        a, f = score_rollout(
            task_type=str(task_types[i] or ""),
            sequence_text=str(seq_texts[i] or ""),
            response_text=str(resp_texts[i] or ""),
            gt=str(gts[i] or ""),
            mc_options_json=str(mc[i] or ""),
            gt_letter=str(gtl[i] or ""),
            answer_style=str(astyle[i] or ""),
            maze_ref_json=str(maze_refs[i] or ""),
            is_valid=iv,
        )
        accs[i] = a
        fids[i] = f

    return select_indices(
        uids=uids,
        accs=accs,
        fids=fids,
        is_valids=is_valids,
        keep=keep,
        fidelity_weight=fidelity_weight,
        seed=seed,
    )
