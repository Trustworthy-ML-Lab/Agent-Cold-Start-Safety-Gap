#!/opt/conda/envs/agent_new/bin/python
"""Extract hidden states for harmful queries only, across depths.

Loads trajectories, reconstructs prompts up to the target harmful turn,
and extracts last-token hidden states. Saves with task IDs for later
joining with judgment labels.

Usage:
    python src/extract_hidden_states_v2.py \
        --model_path gemma4-4b \
        --variant A22 \
        --layer -1 \
        --output hidden_states/gemma4-4b_A22_v2.npz \
        --depths 0,1,3,5,7,10,15,20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import resolve_model_path, apply_chat_template, get_model_family
from envs import ENV_REGISTRY


def load_trajectories(ablation_dir: str, run_idx: int = 0) -> list:
    traj_path = os.path.join(ablation_dir, "trajectories.json")
    with open(traj_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "runs" in raw:
        trajs = raw["runs"][f"run_{run_idx}"]
    else:
        trajs = raw
    return [t for t in trajs if t.get("complete", True)]


def build_prompt_up_to_target(trajectory: dict, tokenizer, model_family: str) -> str | None:
    messages = trajectory["messages"]
    target_idx = trajectory["threat_msg_idx"]
    prompt_messages = messages[:target_idx + 1]

    env_name = trajectory["env"]
    env_cfg = ENV_REGISTRY[env_name]
    env = env_cfg(state={})
    tools_schema = [{"type": "function", "function": d} for d in env.get_tool_descs()]

    try:
        prompt = apply_chat_template(
            tokenizer, prompt_messages, tools=tools_schema, model_family=model_family
        )
        return prompt
    except Exception:
        return None


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--variant", required=True, help="e.g. A1, A21, A22, A23")
    parser.add_argument("--output", required=True, help="Output .npz file")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index (-1 = last)")
    parser.add_argument("--run_idx", type=int, default=0)
    parser.add_argument("--depths", type=str, default="0,1,3,5,7,10,15,20")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoConfig

    model_path = resolve_model_path(args.model_path)
    model_family = get_model_family(model_path)
    depths = [int(d) for d in args.depths.split(",")]

    # Tokenizer
    if model_family == "gemma":
        tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "tokenizer"):
        if tokenizer.tokenizer.pad_token is None:
            tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

    # Load trajectories (harmful only)
    traj_dir = f"eval_results_ablation/{args.model_path}-{args.variant}"
    trajs = load_trajectories(traj_dir, args.run_idx)
    trajs = [t for t in trajs if t["threat_depth"] in depths]
    print(f"Loaded {len(trajs)} trajectories from {traj_dir}")

    # Build prompts
    prompts = []
    metadata = []
    for t in trajs:
        prompt = build_prompt_up_to_target(t, tokenizer, model_family)
        if prompt is not None:
            prompts.append(prompt)
            metadata.append({
                "id": t["id"],
                "env": t["env"],
                "scenario": t["scenario"],
                "depth": t["threat_depth"],
            })

    print(f"Built {len(prompts)} prompts")

    # Model setup
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "num_hidden_layers"):
        num_layers = config.num_hidden_layers
    elif hasattr(config, "text_config"):
        num_layers = config.text_config.num_hidden_layers
    else:
        num_layers = 32

    target_layer = args.layer if args.layer >= 0 else num_layers + args.layer + 1
    print(f"Model: {num_layers} layers, extracting layer {target_layer}")

    print(f"Loading model {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True,
    )
    model.eval()

    if model_family == "gemma" and hasattr(tokenizer, "tokenizer"):
        text_tokenizer = tokenizer.tokenizer
    else:
        text_tokenizer = tokenizer

    device = next(model.parameters()).device

    # Hook to capture last-token hidden state
    captured = {}

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h[:, -1, :].detach().clone()

    # Attach hook
    target_layer_idx = target_layer - 1
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        hook = model.model.language_model.layers[target_layer_idx].register_forward_hook(hook_fn)
        backbone = model.model.language_model
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        hook = model.model.layers[target_layer_idx].register_forward_hook(hook_fn)
        backbone = model.model
    else:
        raise RuntimeError("Cannot find model layers")

    # Extract
    print(f"Extracting hidden states...")
    all_hidden = []

    for i in range(len(prompts)):
        inputs = text_tokenizer(
            prompts[i], return_tensors="pt",
            truncation=True, max_length=32768,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

        with torch.no_grad():
            backbone(input_ids=input_ids, attention_mask=attention_mask)

        h = captured["h"][0].cpu().float().numpy()
        all_hidden.append(h)
        captured.clear()
        del input_ids, attention_mask

        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()
            print(f"  {i + 1}/{len(prompts)}")

    hook.remove()

    # Save
    hidden_array = np.stack(all_hidden)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        hidden_states=hidden_array,
        metadata=json.dumps(metadata),
        model=args.model_path,
        variant=args.variant,
        layer=target_layer,
    )
    print(f"Saved: {args.output} ({hidden_array.shape})")


if __name__ == "__main__":
    main()
