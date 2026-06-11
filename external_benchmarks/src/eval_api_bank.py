"""Evaluate API-Bank results by comparing model's API calls against ground truth.

Compares:
1. API name correctness
2. Parameter correctness (fuzzy matching for string values)

Usage:
    python external_eval/src/eval_api_bank.py \
        --input external_eval/results/qwen3-4b/A1/api_bank/trajectories.json \
        --output external_eval/results/qwen3-4b/A1/api_bank/eval.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def normalize_value(v):
    """Normalize a value for comparison."""
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s


def normalize_time(s: str) -> str:
    """Normalize datetime strings: strip trailing :00, T→space, ignore year."""
    s = s.strip().replace("T", " ")
    # Strip time portion if it's 00:00:00 (date-only comparison)
    s = re.sub(r'\s+00:00:00$', '', s)
    s = re.sub(r':00$', '', s)
    s = re.sub(r':00$', '', s)
    # Strip trailing time if one side is date-only: keep only date part for comparison
    # Done via the caller checking both sides
    # Replace year with wildcard — GT uses 2023 but model may use current year
    s = re.sub(r'^\d{4}', 'YYYY', s)
    return s


def _is_time_field(key: str) -> bool:
    return any(t in key for t in ("time", "date", "check_in", "check_out"))


def _normalize_list_value(v) -> str:
    """Normalize list representations: ['a','b'] vs 'a, b' → sorted comma-joined."""
    s = str(v).strip()
    # Parse Python-style list string
    s = s.strip("[]")
    s = s.replace("'", "").replace('"', '')
    items = [x.strip().lower() for x in s.split(",") if x.strip()]
    return ",".join(sorted(items))


def values_match(key: str, model_val, gt_val) -> bool:
    """Compare a single parameter value with fuzzy matching."""
    norm_m = normalize_value(model_val)
    norm_g = normalize_value(gt_val)

    if norm_m == norm_g:
        return True

    # Numeric comparison
    try:
        if float(model_val) == float(gt_val):
            return True
    except (ValueError, TypeError):
        pass

    # Time normalization (ignore year, format diffs)
    if _is_time_field(key):
        nt_m = normalize_time(norm_m)
        nt_g = normalize_time(norm_g)
        if nt_m == nt_g:
            return True
        # Date-only match: if one is prefix of the other (date vs datetime)
        if nt_m.startswith(nt_g) or nt_g.startswith(nt_m):
            return True

    # List comparison (attendees: "John, Jane" vs "['John', 'Jane']")
    if "attendee" in key or str(gt_val).startswith("["):
        if _normalize_list_value(model_val) == _normalize_list_value(gt_val):
            return True

    # Containment: if GT is substring of model or vice versa
    if len(norm_g) > 3 and (norm_g in norm_m or norm_m in norm_g):
        return True

    return False


def params_match(model_args: dict, gt_params: dict, strict: bool = False) -> bool:
    """Check if model arguments match ground truth parameters."""
    if strict:
        return model_args == gt_params

    for key, gt_val in gt_params.items():
        if key == "token":
            continue
        model_val = model_args.get(key)
        if model_val is None:
            return False
        if not values_match(key, model_val, gt_val):
            return False
    return True


# Auth APIs that are intermediate steps — models often skip these
SKIP_APIS = {"GetUserToken", "CheckToken"}


def evaluate_segment(model_calls: list, gt_calls: list) -> dict:
    """Evaluate one segment's model calls against GT calls.

    Skips auth-only APIs (GetUserToken) since models reasonably skip them.
    Returns dict with name_correct, param_correct, total_gt.
    """
    # Filter out auth-only GT calls
    gt_calls = [g for g in gt_calls if g["api_name"] not in SKIP_APIS]
    if not gt_calls:
        return {"name_correct": 0, "param_correct": 0, "total_gt": 0}

    name_correct = 0
    param_correct = 0
    matched = set()

    for gt in gt_calls:
        gt_name = gt["api_name"]
        gt_params = gt.get("param_dict", {})

        for i, mc in enumerate(model_calls):
            if i in matched:
                continue
            if mc["api_name"] == gt_name:
                name_correct += 1
                if params_match(mc.get("arguments", {}), gt_params):
                    param_correct += 1
                matched.add(i)
                break

    return {
        "name_correct": name_correct,
        "param_correct": param_correct,
        "total_gt": len(gt_calls),
    }


def evaluate_task(model_segments: list, gt_segments: list) -> dict:
    """Evaluate a task by matching segments pairwise.

    Both model_segments and gt_segments are list-of-lists.
    Returns aggregate metrics across all segments.
    """
    total_gt = 0
    total_name = 0
    total_param = 0

    for seg_idx, gt_seg in enumerate(gt_segments):
        model_seg = model_segments[seg_idx] if seg_idx < len(model_segments) else []
        res = evaluate_segment(model_seg, gt_seg)
        total_gt += res["total_gt"]
        total_name += res["name_correct"]
        total_param += res["param_correct"]

    accuracy = total_param / total_gt if total_gt > 0 else 1.0
    return {
        "api_accuracy": accuracy,
        "name_correct": total_name,
        "param_correct": total_param,
        "total_gt": total_gt,
        "total_model_segments": len(model_segments),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    runs = data["runs"]
    by_depth = defaultdict(lambda: {"total": 0, "name_acc": 0, "param_acc": 0,
                                     "total_gt_calls": 0, "total_name_correct": 0,
                                     "total_param_correct": 0})

    for run_key, run_data in runs.items():
        for depth_str, tasks in run_data.items():
            depth = int(depth_str)
            for task in tasks:
                gt_segments = task.get("gt_api_calls", [])
                model_segments = task.get("model_api_calls", [])

                result = evaluate_task(model_segments, gt_segments)
                by_depth[depth]["total"] += 1
                by_depth[depth]["name_acc"] += result["name_correct"] / max(result["total_gt"], 1)
                by_depth[depth]["param_acc"] += result["api_accuracy"]
                by_depth[depth]["total_gt_calls"] += result["total_gt"]
                by_depth[depth]["total_name_correct"] += result["name_correct"]
                by_depth[depth]["total_param_correct"] += result["param_correct"]

    eval_result = {"by_depth": {}}
    agg_gt = 0
    agg_name = 0
    agg_param = 0
    total_tasks = 0

    for depth in sorted(by_depth.keys()):
        info = by_depth[depth]
        n = info["total"]
        total_tasks += n
        agg_gt += info["total_gt_calls"]
        agg_name += info["total_name_correct"]
        agg_param += info["total_param_correct"]
        eval_result["by_depth"][str(depth)] = {
            "total_tasks": n,
            "total_gt_calls": info["total_gt_calls"],
            "name_accuracy": info["total_name_correct"] / info["total_gt_calls"] if info["total_gt_calls"] > 0 else 0,
            "param_accuracy": info["total_param_correct"] / info["total_gt_calls"] if info["total_gt_calls"] > 0 else 0,
            "avg_task_name_acc": info["name_acc"] / n if n > 0 else 0,
            "avg_task_param_acc": info["param_acc"] / n if n > 0 else 0,
        }

    eval_result["n_total_tasks"] = total_tasks
    eval_result["n_total_gt_calls"] = agg_gt
    eval_result["name_accuracy"] = agg_name / agg_gt if agg_gt > 0 else 0
    eval_result["param_accuracy"] = agg_param / agg_gt if agg_gt > 0 else 0

    print(f"API-Bank Results ({total_tasks} tasks, {agg_gt} GT calls)")
    print(f"  Accuracy (name+args): {eval_result['param_accuracy']*100:.1f}%")
    print(f"  Name-only Accuracy:   {eval_result['name_accuracy']*100:.1f}%")
    print()
    print("By depth:")
    for depth in sorted(by_depth.keys()):
        info = eval_result["by_depth"][str(depth)]
        print(f"  d={depth}: {info['param_accuracy']*100:.1f}%  (name={info['name_accuracy']*100:.1f}%)  [{info['total_tasks']} tasks]")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
