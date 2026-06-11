"""Evaluate BFCL accuracy using state-based comparison.

Compares model's tool calls against ground truth by executing both
on fresh environment instances and comparing final state.

Usage:
    python external_eval/src/eval_bfcl.py \
        --input external_eval/results/qwen3-4b/bfcl_d0.json \
        --output external_eval/results/qwen3-4b/eval_bfcl_d0.json
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import re
import sys
from pathlib import Path

EXTERNAL_EVAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXTERNAL_EVAL_ROOT / "bfcl" / "environments"))

CLASS_TO_MODULE = {
    "GorillaFileSystem": "gorilla_file_system", "VehicleControlAPI": "vehicle_control",
    "TradingBot": "trading_bot", "TravelAPI": "travel_booking",
    "TicketAPI": "ticket_api", "MessageAPI": "message_api",
    "TwitterAPI": "posting_api", "MathAPI": "math_api", "MemoryAPI_kv": "memory_kv",
}
UNSUPPORTED = {"WebSearchAPI", "MemoryAPI_rec_sum", "MemoryAPI_vector"}


def create_instances(involved_classes, initial_config):
    instances = {}
    for cls_name in involved_classes:
        if cls_name in UNSUPPORTED:
            continue
        mod_name = CLASS_TO_MODULE.get(cls_name)
        if not mod_name:
            continue
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            inst = cls()
            cfg = initial_config.get(cls_name, {})
            if cfg and hasattr(inst, "_load_scenario"):
                inst._load_scenario(copy.deepcopy(cfg))
            instances[cls_name] = inst
        except Exception:
            pass
    return instances


def execute_gt_calls(instances, gt_turns):
    method_map = {}
    for cls_name, inst in instances.items():
        for attr_name in dir(inst):
            if attr_name.startswith("_"):
                continue
            if callable(getattr(inst, attr_name)):
                method_map[attr_name] = inst

    for turn_calls in gt_turns:
        for call_str in turn_calls:
            try:
                func_name = call_str.split("(")[0].strip()
                if func_name in method_map:
                    inst = method_map[func_name]
                    eval(f"inst.{call_str}")
            except Exception:
                pass


def extract_model_calls(result):
    calls = []
    for msg in result.get("messages", []):
        if msg["role"] != "assistant":
            continue
        content = msg.get("content", "")
        if "<tool_call>" in content:
            # Try JSON format first: <tool_call>{"name":...}</tool_call>
            found_json = False
            for m in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL):
                try:
                    parsed = json.loads(m.group(1))
                    calls.append((parsed.get("name", ""), parsed.get("arguments", parsed.get("parameters", {}))))
                    found_json = True
                except json.JSONDecodeError:
                    pass
            # Try Qwen 3.5 XML format: <tool_call><function=name><parameter=arg>val</parameter></function></tool_call>
            if not found_json and "<function=" in content:
                for fn_match in re.finditer(r'<function=([\w.]+)>(.*?)</function>', content, re.DOTALL):
                    name = fn_match.group(1)
                    params_str = fn_match.group(2)
                    args = {}
                    for pm in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', params_str, re.DOTALL):
                        key = pm.group(1)
                        val = pm.group(2).strip()
                        try:
                            args[key] = json.loads(val)
                        except (json.JSONDecodeError, ValueError):
                            args[key] = val
                    calls.append((name, args))
        elif "call:" in content:
            # Gemma format: call:func_name{key:value,key:value}
            for m in re.finditer(r'call:(\w+)\{(.*?)\}', content, re.DOTALL):
                name = m.group(1)
                raw_args = m.group(2).strip().replace('<|"|>', '"')
                args = {}
                # Try JSON first
                try:
                    args = json.loads("{" + raw_args + "}")
                except (json.JSONDecodeError, ValueError):
                    # Parse key:value pairs manually
                    for kv in re.finditer(r'(\w+)\s*:\s*(.+?)(?:,\s*(?=\w+\s*:)|$)', raw_args):
                        key = kv.group(1)
                        val = kv.group(2).strip()
                        # Try to parse as number/bool
                        if val.isdigit():
                            args[key] = int(val)
                        elif val.lower() in ('true', 'false'):
                            args[key] = val.lower() == 'true'
                        else:
                            args[key] = val
                calls.append((name, args))
        else:
            # Try full content as JSON first
            try:
                parsed = json.loads(content.strip())
                if isinstance(parsed, dict) and "name" in parsed:
                    calls.append((parsed["name"], parsed.get("arguments", parsed.get("parameters", {}))))
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
            # Try semicolon-separated JSON objects (e.g. Llama native format)
            if ";" in content and '"name"' in content:
                found_semi = False
                for part in content.split(";"):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        parsed = json.loads(part)
                        if isinstance(parsed, dict) and "name" in parsed:
                            calls.append((parsed["name"], parsed.get("arguments", parsed.get("parameters", {}))))
                            found_semi = True
                    except (json.JSONDecodeError, ValueError):
                        pass
                if found_semi:
                    continue
            # Search for JSON objects within text (fallback)
            for m in re.finditer(r'\{[^{}]*"name"[^{}]*\}', content):
                try:
                    parsed = json.loads(m.group(0))
                    if isinstance(parsed, dict) and "name" in parsed:
                        calls.append((parsed["name"], parsed.get("arguments", parsed.get("parameters", {}))))
                except (json.JSONDecodeError, ValueError):
                    pass
    return calls


def execute_model_calls(instances, calls):
    method_map = {}
    for cls_name, inst in instances.items():
        for attr_name in dir(inst):
            if attr_name.startswith("_"):
                continue
            if callable(getattr(inst, attr_name)):
                method_map[attr_name] = inst

    for func_name, args in calls:
        if func_name in method_map:
            try:
                getattr(method_map[func_name], func_name)(**args)
            except Exception:
                pass


def compare_states(model_instances, gt_instances):
    for cls_name, gt_inst in gt_instances.items():
        if cls_name not in model_instances:
            return False
        model_inst = model_instances[cls_name]
        gt_attrs = {k: v for k, v in vars(gt_inst).items() if not k.startswith("_")}
        model_attrs = {k: v for k, v in vars(model_inst).items() if not k.startswith("_")}
        if gt_attrs != model_attrs:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default=str(EXTERNAL_EVAL_ROOT / "data" / "bfcl" / "BFCL_v4_multi_turn_base.json"))
    parser.add_argument("--gt", default=str(EXTERNAL_EVAL_ROOT / "data" / "bfcl" / "BFCL_v4_multi_turn_base_gt.json"))
    args = parser.parse_args()

    with open(args.input) as f:
        raw = json.load(f)

    # Handle multi-run format
    if isinstance(raw, dict) and "runs" in raw:
        results = []
        for run_key in sorted(raw["runs"].keys()):
            results.extend(raw["runs"][run_key])
    else:
        results = raw

    tasks = [json.loads(l) for l in open(args.data) if l.strip()]
    task_map = {t["id"]: t for t in tasks}
    gt_map = {}
    for line in open(args.gt):
        if line.strip():
            entry = json.loads(line)
            gt_map[entry["id"]] = entry["ground_truth"]

    n_correct, n_incorrect, n_error = 0, 0, 0
    per_task = []

    for r in results:
        tid = r["task_id"]
        task = task_map.get(tid, {})
        gt_turns = gt_map.get(tid, [])

        involved = task.get("involved_classes", [])
        config = task.get("initial_config", {})

        try:
            gt_inst = create_instances(involved, config)
            execute_gt_calls(gt_inst, gt_turns)

            model_inst = create_instances(involved, config)
            calls = extract_model_calls(r)
            execute_model_calls(model_inst, calls)

            correct = compare_states(model_inst, gt_inst)
        except Exception:
            correct = False
            n_error += 1

        if correct:
            n_correct += 1
        else:
            n_incorrect += 1

        per_task.append({"task_id": tid, "correct": correct, "depth": r.get("depth", 0), "run_idx": r.get("run_idx", 0)})

    n_total = len(results)

    # Per-depth per-run breakdown for mean ± std
    import numpy as np
    from collections import defaultdict

    by_depth_run = defaultdict(lambda: {"correct": 0, "total": 0})
    for pt in per_task:
        d = pt["depth"]
        run = pt["run_idx"]
        by_depth_run[(d, run)]["total"] += 1
        if pt["correct"]:
            by_depth_run[(d, run)]["correct"] += 1

    all_depths = sorted(set(d for d, _ in by_depth_run.keys()))
    all_runs = sorted(set(r for _, r in by_depth_run.keys()))

    by_depth_stats = {}
    for d in all_depths:
        run_accs = []
        total_correct = 0
        total_all = 0
        for run in all_runs:
            s = by_depth_run.get((d, run), {"correct": 0, "total": 0})
            if s["total"] > 0:
                run_accs.append(s["correct"] / s["total"])
                total_correct += s["correct"]
                total_all += s["total"]
        by_depth_stats[str(d)] = {
            "total": total_all,
            "correct": total_correct,
            "accuracy": round(total_correct / total_all, 4) if total_all else 0,
            "mean": round(float(np.mean(run_accs)), 4) if run_accs else 0,
            "std": round(float(np.std(run_accs)), 4) if run_accs else 0,
            "n_runs": len(run_accs),
        }

    metrics = {
        "n_total": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_error": n_error,
        "accuracy": round(n_correct / n_total, 4) if n_total else 0,
        "by_depth": by_depth_stats,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"BFCL Accuracy: {n_correct}/{n_total} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Errors: {n_error}")
    print(f"\nBy depth (mean ± std across {len(all_runs)} runs):")
    for d in all_depths:
        s = by_depth_stats[str(d)]
        print(f"  d={d}: {s['mean']*100:.1f}% ± {s['std']*100:.1f}%")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
