"""Pack trajectories + judgments into organized output dirs.

Cold-start study: user-threats only, depths [0, 1, 3, 5, 7, 10].

Usage:
    python src/pack_results.py --eval_dir eval_results/qwen2.5-7b
"""
import argparse
import json
import os
from collections import defaultdict


DEPTHS = [0, 1, 3, 5, 7, 10, 15, 20]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True)
    args = parser.parse_args()

    traj_path = os.path.join(args.eval_dir, "trajectories.json")
    judg_path = os.path.join(args.eval_dir, "judgments.json")

    with open(traj_path) as f:
        raw_trajs = json.load(f)
    with open(judg_path) as f:
        judg = json.load(f)

    # Handle multi-run format
    if isinstance(raw_trajs, dict) and "runs" in raw_trajs:
        trajs = []
        for run_key in sorted(raw_trajs["runs"].keys()):
            trajs.extend(raw_trajs["runs"][run_key])
        n_runs = raw_trajs["n_runs"]
    else:
        trajs = raw_trajs
        n_runs = 1

    # Filter out incomplete tasks
    n_before = len(trajs)
    trajs = [t for t in trajs if t.get("complete", True)]
    n_skipped = n_before - len(trajs)
    if n_skipped:
        print(f"  Skipped {n_skipped} incomplete tasks")

    jmap = {j["id"]: j for j in judg["judgments"]}

    for t in trajs:
        uid = f"{t['id']}_r{t.get('run_idx', 0)}"
        j = jmap.get(uid, {})
        t["_verdict"] = j.get("verdict", "ERROR")
        t["_reason"] = j.get("reason", "")

    # --- by_env ---
    by_env_dir = os.path.join(args.eval_dir, "by_env")
    os.makedirs(by_env_dir, exist_ok=True)
    by_env = defaultdict(list)
    for t in trajs:
        by_env[t.get("env", "unknown")].append(t)
    for env, items in by_env.items():
        env_dir = os.path.join(by_env_dir, env)
        os.makedirs(env_dir, exist_ok=True)
        with open(os.path.join(env_dir, "all.json"), "w") as f:
            json.dump(items, f, indent=2, ensure_ascii=False, default=str)
        safe_items = [t for t in items if t["_verdict"] == "SAFE"]
        unsafe_items = [t for t in items if t["_verdict"] == "UNSAFE"]
        if safe_items:
            with open(os.path.join(env_dir, "safe.json"), "w") as f:
                json.dump(safe_items, f, indent=2, ensure_ascii=False, default=str)
        if unsafe_items:
            with open(os.path.join(env_dir, "unsafe.json"), "w") as f:
                json.dump(unsafe_items, f, indent=2, ensure_ascii=False, default=str)

    # --- by_depth ---
    by_depth_dir = os.path.join(args.eval_dir, "by_depth")
    os.makedirs(by_depth_dir, exist_ok=True)
    for d in DEPTHS:
        d_items = [t for t in trajs if t["threat_depth"] == d]
        if not d_items:
            continue
        d_dir = os.path.join(by_depth_dir, f"d{d}")
        os.makedirs(d_dir, exist_ok=True)
        with open(os.path.join(d_dir, "all.json"), "w") as f:
            json.dump(d_items, f, indent=2, ensure_ascii=False, default=str)
        safe_d = [t for t in d_items if t["_verdict"] == "SAFE"]
        unsafe_d = [t for t in d_items if t["_verdict"] == "UNSAFE"]
        if safe_d:
            with open(os.path.join(d_dir, "safe.json"), "w") as f:
                json.dump(safe_d, f, indent=2, ensure_ascii=False, default=str)
        if unsafe_d:
            with open(os.path.join(d_dir, "unsafe.json"), "w") as f:
                json.dump(unsafe_d, f, indent=2, ensure_ascii=False, default=str)

    # --- Stats ---
    safe_all = [t for t in trajs if t["_verdict"] == "SAFE"]
    unsafe_all = [t for t in trajs if t["_verdict"] == "UNSAFE"]

    stats = {
        "total_tasks": len(trajs),
        "total_safe": len(safe_all),
        "total_unsafe": len(unsafe_all),
        "overall_safety_rate": round(len(safe_all) / len(trajs), 4) if trajs else 0,
        "by_depth": {},
        "by_env": {},
        "by_env_depth": {},
    }

    for d in DEPTHS:
        d_tasks = [t for t in trajs if t["threat_depth"] == d]
        d_safe = sum(1 for t in d_tasks if t["_verdict"] == "SAFE")
        stats["by_depth"][str(d)] = {
            "total": len(d_tasks),
            "safe": d_safe,
            "unsafe": len(d_tasks) - d_safe,
            "safety_rate": round(d_safe / len(d_tasks), 4) if d_tasks else 0,
        }

    for env in sorted(by_env.keys()):
        env_tasks = by_env[env]
        env_safe = sum(1 for t in env_tasks if t["_verdict"] == "SAFE")
        stats["by_env"][env] = {
            "total": len(env_tasks),
            "safe": env_safe,
            "unsafe": len(env_tasks) - env_safe,
            "safety_rate": round(env_safe / len(env_tasks), 4) if env_tasks else 0,
        }
        stats["by_env_depth"][env] = {}
        for d in DEPTHS:
            ed_tasks = [t for t in env_tasks if t["threat_depth"] == d]
            ed_safe = sum(1 for t in ed_tasks if t["_verdict"] == "SAFE")
            stats["by_env_depth"][env][str(d)] = {
                "total": len(ed_tasks),
                "safe": ed_safe,
                "safety_rate": round(ed_safe / len(ed_tasks), 4) if ed_tasks else 0,
            }

    stats_path = os.path.join(args.eval_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # --- Compute per-run stats for mean ± std (only if n_runs > 1) ---
    import numpy as np

    if n_runs > 1:
        # Compute safety rate PER RUN, then report mean ± std across runs
        # Group by run_idx
        by_run = defaultdict(list)
        for t in trajs:
            by_run[t.get("run_idx", 0)].append(t)

        # Per-depth: compute safety rate for each run, then mean/std across runs
        depth_trial_stats = {}
        for d in DEPTHS:
            run_rates = []
            for run_idx in sorted(by_run.keys()):
                run_trajs = by_run[run_idx]
                d_trajs = [t for t in run_trajs if t["threat_depth"] == d]
                if d_trajs:
                    rate = sum(1 for t in d_trajs if t["_verdict"] == "SAFE") / len(d_trajs)
                    run_rates.append(rate)
            if run_rates:
                depth_trial_stats[d] = {
                    "mean": float(np.mean(run_rates)),
                    "std": float(np.std(run_rates)),
                    "n_runs": len(run_rates),
                }

        # Per-env per-depth
        env_depth_trial_stats = defaultdict(dict)
        for d in DEPTHS:
            for env in sorted(by_env.keys()):
                run_rates = []
                for run_idx in sorted(by_run.keys()):
                    run_trajs = by_run[run_idx]
                    ed_trajs = [t for t in run_trajs if t["threat_depth"] == d and t.get("env") == env]
                    if ed_trajs:
                        rate = sum(1 for t in ed_trajs if t["_verdict"] == "SAFE") / len(ed_trajs)
                        run_rates.append(rate)
                if run_rates:
                    env_depth_trial_stats[env][d] = {
                        "mean": float(np.mean(run_rates)),
                        "std": float(np.std(run_rates)),
                    }

        stats["by_depth_with_std"] = {str(d): depth_trial_stats.get(d, {}) for d in DEPTHS}
        stats["by_env_depth_with_std"] = {env: {str(d): v for d, v in depths.items()} for env, depths in env_depth_trial_stats.items()}

    stats_path = os.path.join(args.eval_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # --- Print summary ---
    print(f"\nPacked {len(trajs)} trajectories (n_runs={n_runs})")
    print(f"  SAFE: {len(safe_all)} | UNSAFE: {len(unsafe_all)} | Overall: {stats['overall_safety_rate']*100:.1f}%")

    print(f"\n=== Safety Rate by Depth ===")
    if n_runs > 1:
        print(f"{'Depth':<10}{'Rate':<12}{'Std':<10}{'N':<8}")
        print("-" * 40)
        for d in DEPTHS:
            s = depth_trial_stats.get(d, {})
            if s:
                print(f"d={d:<7}{s['mean']*100:5.1f}%     ±{s['std']*100:4.1f}%    {s['n_runs']}")
    else:
        print(f"{'Depth':<10}{'Rate':<12}{'Safe':<8}{'Total':<8}")
        print("-" * 38)
        for d in DEPTHS:
            s = stats["by_depth"].get(str(d), {})
            if s and s["total"] > 0:
                print(f"d={d:<7}{s['safety_rate']*100:5.1f}%     {s['safe']:<8}{s['total']:<8}")

    print(f"\n=== Safety Rate by Env ===")
    print(f"{'Env':<25}{'Rate':<10}{'Safe':<8}{'Total':<8}")
    print("-" * 51)
    for env in sorted(by_env.keys()):
        s = stats["by_env"][env]
        print(f"{env:<25}{s['safety_rate']*100:5.1f}%    {s['safe']:<8}{s['total']:<8}")

    print(f"\n=== Safety Rate: Env × Depth ===")
    print(f"{'Env':<20}" + "".join(f"{'D='+str(d):<10}" for d in DEPTHS))
    print("-" * (20 + 10 * len(DEPTHS)))
    for env in sorted(by_env.keys()):
        row = f"{env:<20}"
        for d in DEPTHS:
            s = stats["by_env_depth"].get(env, {}).get(str(d), {})
            if s and s.get("total", 0) > 0:
                row += f"{s['safety_rate']*100:5.1f}%    "
            else:
                row += f"{'N/A':<10}"
        print(row)

    print(f"\nSaved stats to {stats_path}")

    # --- Generate markdown report ---
    md_path = os.path.join(args.eval_dir, "report.md")
    lines = []
    lines.append(f"# Safety Evaluation Report\n")
    lines.append(f"- **Total tasks**: {len(trajs)}")
    lines.append(f"- **Runs**: {n_runs}")
    lines.append(f"- **SAFE**: {len(safe_all)} | **UNSAFE**: {len(unsafe_all)}")
    lines.append(f"- **Overall safety rate**: {stats['overall_safety_rate']*100:.1f}%\n")

    # Depth table
    lines.append("## Safety Rate by Depth\n")
    if n_runs > 1:
        lines.append("| Depth | Rate | Std | Runs |")
        lines.append("|-------|------|-----|------|")
        for d in DEPTHS:
            s = depth_trial_stats.get(d, {})
            if s:
                lines.append(f"| d={d} | {s['mean']*100:.1f}% | ±{s['std']*100:.1f}% | {s['n_runs']} |")
    else:
        lines.append("| Depth | Rate | Safe | Total |")
        lines.append("|-------|------|------|-------|")
        for d in DEPTHS:
            s = stats["by_depth"].get(str(d), {})
            if s and s["total"] > 0:
                lines.append(f"| d={d} | {s['safety_rate']*100:.1f}% | {s['safe']} | {s['total']} |")

    # Env table
    lines.append("\n## Safety Rate by Env\n")
    lines.append("| Env | Rate | Safe | Total |")
    lines.append("|-----|------|------|-------|")
    for env in sorted(by_env.keys()):
        s = stats["by_env"][env]
        lines.append(f"| {env} | {s['safety_rate']*100:.1f}% | {s['safe']} | {s['total']} |")

    # Env × Depth table
    lines.append("\n## Safety Rate: Env × Depth\n")
    header = "| Env | " + " | ".join(f"D={d}" for d in DEPTHS) + " |"
    sep = "|-----|" + "|".join("------" for _ in DEPTHS) + "|"
    lines.append(header)
    lines.append(sep)
    for env in sorted(by_env.keys()):
        row = f"| {env} |"
        for d in DEPTHS:
            s = stats["by_env_depth"].get(env, {}).get(str(d), {})
            if s and s.get("total", 0) > 0:
                row += f" {s['safety_rate']*100:.1f}% |"
            else:
                row += " N/A |"
        lines.append(row)

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved report to {md_path}")


if __name__ == "__main__":
    main()
