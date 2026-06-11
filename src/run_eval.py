"""Run depth-controlled safety evaluation.

Loads tasks from data/tasks.json, runs multi-turn agent loop with vLLM,
saves trajectories for judging.

Usage:
    python src/run_eval.py --model_path qwen2.5-7b --output_dir eval_results/qwen2.5-7b
    python src/run_eval.py --model_path qwen2.5-7b --tasks data/tasks.json --tensor_parallel_size 4
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    resolve_model_path, classify_output, classify_output_multi,
    apply_chat_template, load_generation_config, is_native_tool_format,
    get_model_family, AGENT_SYSTEM,
)
from envs import ENV_REGISTRY


def run_eval(tasks, llm, tokenizer, sampling_params, model_path, max_model_len, max_rounds_per_turn=5, processor=None):
    from tqdm import tqdm
    native = is_native_tool_format(model_path)
    model_family = get_model_family(model_path)

    states = []
    for t in tqdm(tasks, desc="Init"):
        env_cfg = t["environments"][0]
        env = ENV_REGISTRY[env_cfg["name"]](state=copy.deepcopy(env_cfg["parameters"]))
        tools_schema = [{"type": "function", "function": d} for d in env.get_tool_descs()]

        messages = [{"role": "system", "content": t.get("system_prompt", AGENT_SYSTEM)}]

        states.append({
            "task": t,
            "env": env,
            "tools_schema": tools_schema,
            "messages": messages,
            "user_turn_idx": 0,
            "phase": "need_user_turn",
            "done": False,
            "rounds": 0,
            "rounds_this_turn": 0,
            "trace": [],
        })

    r = 0
    while True:
        r += 1
        for s in states:
            if s["done"] or s["phase"] != "need_user_turn":
                continue
            user_turns = s["task"]["user_turns"]
            if s["user_turn_idx"] >= len(user_turns):
                s["done"] = True
                continue
            s["messages"].append({"role": "user", "content": user_turns[s["user_turn_idx"]]})
            if s["user_turn_idx"] == len(user_turns) - 1:
                s["threat_msg_idx"] = len(s["messages"]) - 1
            s["user_turn_idx"] += 1
            s["phase"] = "wait_model"
            s["rounds_this_turn"] = 0

        active = [i for i, s in enumerate(states)
                  if not s["done"] and s["phase"] in ("wait_model", "wait_model_after_tool")]
        if not active:
            break

        prompts, valid = [], []
        for i in active:
            s = states[i]
            try:
                prompt = apply_chat_template(tokenizer, s["messages"], tools=s["tools_schema"], model_family=model_family)
                prompts.append(prompt)
                valid.append(i)
            except Exception as e:
                if s["rounds"] == 0:
                    print(f"  Template error on {s['task']['id']}: {e}")
                s["done"] = True
                continue

        if not prompts:
            done_count = sum(1 for s in states if s["done"])
            print(f"  Round {r}: {done_count}/{len(states)} done (no prompts)")
            continue

        try:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=(len(prompts) > 50))
        except Exception as e:
            if "context length" in str(e).lower() or "input_tokens" in str(e).lower():
                print(f"  Round {r}: context length exceeded, marking {len(valid)} tasks done")
                for i in valid:
                    states[i]["done"] = True
                continue
            raise

        for idx, output in zip(valid, outputs):
            s = states[idx]
            gen = output.outputs[0].text if output.outputs else ""
            out = classify_output_multi(gen, model_family=model_family, processor=processor)
            s["rounds"] += 1
            s["rounds_this_turn"] += 1

            if out["type"] == "tool":
                tool_calls = out["tool_calls"]

                if model_family == "gemma":
                    # Gemma: assistant message needs structured tool_calls field
                    # for the chat template to pair with subsequent tool responses
                    s["messages"].append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": tc["name"], "arguments": tc.get("arguments", {})}}
                            for tc in tool_calls
                        ],
                    })
                else:
                    s["messages"].append({"role": "assistant", "content": gen})

                all_results = []
                for tc in tool_calls:
                    result = s["env"].call_tool(tc["name"], tc.get("arguments", {}))
                    all_results.append(json.dumps(result, default=str, ensure_ascii=False))
                    s["trace"].append({
                        "round": s["rounds"],
                        "type": "tool_call",
                        "tool_name": tc["name"],
                    })

                if model_family == "gemma":
                    # Gemma: each tool response as a separate role "tool" message
                    for i, (tc, res) in enumerate(zip(tool_calls, all_results)):
                        s["messages"].append({
                            "role": "tool",
                            "name": tc["name"],
                            "content": res,
                        })
                elif native:
                    tool_content = "\n".join(all_results) if len(all_results) > 1 else all_results[0]
                    s["messages"].append({"role": "tool", "content": tool_content})
                else:
                    tool_content = "\n".join(all_results) if len(all_results) > 1 else all_results[0]
                    s["messages"].append({
                        "role": "user",
                        "content": f"<tool_response>\n{tool_content}\n</tool_response>"
                    })
                s["phase"] = "wait_model_after_tool"
            else:
                s["messages"].append({"role": "assistant", "content": gen})
                s["trace"].append({"round": s["rounds"], "type": "text_response"})
                s["phase"] = "need_user_turn"

            if s["rounds_this_turn"] >= max_rounds_per_turn:
                s["phase"] = "need_user_turn"

        done_count = sum(1 for s in states if s["done"])
        print(f"  Round {r}: active={len(valid)} done={done_count}/{len(states)}")

    results = []
    for s in states:
        total_user_turns = len(s["task"]["user_turns"])
        completed = s["user_turn_idx"] >= total_user_turns
        results.append({
            "id": s["task"]["id"],
            "env": s["task"]["environments"][0]["name"],
            "scenario": s["task"].get("scenario", ""),
            "threat_source": s["task"]["threat_source"],
            "threat_depth": s["task"]["threat_depth"],
            "harm_description": s["task"].get("harm_description", ""),
            "messages": s["messages"],
            "trace": s["trace"],
            "rounds": s["rounds"],
            "user_turns_completed": s["user_turn_idx"],
            "threat_msg_idx": s.get("threat_msg_idx"),
            "complete": completed,
        })
    n_incomplete = sum(1 for r in results if not r["complete"])
    if n_incomplete:
        print(f"  WARNING: {n_incomplete}/{len(results)} tasks incomplete (context length exceeded)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tasks", default="data/tasks.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_model_len", type=int, default=0,
                        help="Max context length. 0 = auto-detect from model config.")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_rounds_per_turn", type=int, default=5)
    parser.add_argument("--n_runs", type=int, default=3,
                        help="Number of repeated runs for std estimation")
    args = parser.parse_args()

    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_path = resolve_model_path(args.model_path)
    model_family = get_model_family(model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    processor = None
    if model_family == "gemma":
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    gen_params = load_generation_config(model_path)
    print(f"Model family: {model_family}")
    print(f"Generation config: {gen_params}")

    max_model_len = args.max_model_len
    if max_model_len == 0:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        max_model_len = getattr(config, "max_position_embeddings", None)
        if max_model_len is None:
            text_cfg = getattr(config, "text_config", None)
            if text_cfg:
                max_model_len = getattr(text_cfg, "max_position_embeddings", 32768)
            else:
                max_model_len = 32768
        # Cap at 131072 to avoid OOM on smaller GPUs
        max_model_len = min(max_model_len, 131072)
    print(f"Max model length: {max_model_len}")

    llm = LLM(model=model_path, tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=max_model_len, dtype="bfloat16", trust_remote_code=True)

    sp_kwargs = {"max_tokens": args.max_new_tokens}
    if "temperature" in gen_params:
        sp_kwargs["temperature"] = gen_params["temperature"]
    if "top_p" in gen_params:
        sp_kwargs["top_p"] = gen_params["top_p"]
    if "top_k" in gen_params:
        sp_kwargs["top_k"] = gen_params["top_k"]
    sampling_params = SamplingParams(**sp_kwargs)

    from utils import load_tasks
    tasks = load_tasks(args.tasks)
    print(f"Running {args.n_runs} trials...")

    all_results = {}
    for run_idx in range(args.n_runs):
        print(f"\n{'='*60}")
        print(f"  Run {run_idx + 1}/{args.n_runs}")
        print(f"{'='*60}")

        results = run_eval(tasks, llm, tokenizer, sampling_params, model_path,
                           max_model_len, args.max_rounds_per_turn, processor=processor)

        # Tag each result with run index
        for r in results:
            r["run_idx"] = run_idx

        all_results[f"run_{run_idx}"] = results

    # Save all runs in one file
    traj_path = os.path.join(args.output_dir, "trajectories.json")
    save_data = {
        "n_runs": args.n_runs,
        "n_tasks": len(tasks),
        "runs": all_results,
    }
    with open(traj_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved {len(tasks)} tasks × {args.n_runs} runs to {traj_path}")

    from collections import Counter
    first_run = all_results["run_0"]
    by_depth = Counter(r["threat_depth"] for r in first_run)
    avg_rounds = sum(r["rounds"] for r in first_run) / len(first_run) if first_run else 0
    print(f"By depth: {dict(sorted(by_depth.items()))}")
    print(f"Avg rounds: {avg_rounds:.1f}")


if __name__ == "__main__":
    main()
