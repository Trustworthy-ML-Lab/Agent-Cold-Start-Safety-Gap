#!/opt/conda/envs/agent_new/bin/python
"""Train a linear safety probe on hidden states and save to disk.

Trains logistic regression on ALL available variant data for a model,
saves the probe + scaler for reuse in visualization.

Usage:
    python src/train_safety_probe.py \
        --model llama3.1-8b \
        --variants A1 A21 A22 A23 \
        --output hidden_states/probe_llama3.1-8b.pkl
"""
import argparse
import json
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_hidden_states(path: str):
    data = np.load(path, allow_pickle=True)
    hidden = data["hidden_states"]
    meta_raw = str(data["metadata"])
    meta = json.loads(meta_raw)
    return hidden, meta


def load_judgments(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    judgments = data["judgments"]
    lookup = {}
    for j in judgments:
        base_id = j["id"].rsplit("_r", 1)[0]
        lookup[base_id] = j["verdict"]
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--variants", nargs="+", default=["A1", "A21", "A22", "A23"])
    parser.add_argument("--output", required=True, help="Output .pkl file")
    args = parser.parse_args()

    all_hidden = []
    all_labels = []

    for v in args.variants:
        npz_path = f"hidden_states/{args.model}_{v}_v2.npz"
        jdg_path = f"eval_results_ablation/{args.model}-{v}/judgments.json"

        try:
            hidden, meta = load_hidden_states(npz_path)
            verdicts = load_judgments(jdg_path)
        except FileNotFoundError as e:
            print(f"  Skip {v}: {e}")
            continue

        labels = [verdicts.get(m["id"], "UNKNOWN") for m in meta]
        all_hidden.append(hidden)
        all_labels.extend(labels)
        print(f"  {v}: {len(meta)} samples")

    all_hidden = np.concatenate(all_hidden, axis=0)
    all_labels = np.array(all_labels)

    known_mask = all_labels != "UNKNOWN"
    binary_labels = (all_labels[known_mask] == "SAFE").astype(int)

    print(f"\nTotal samples: {len(all_labels)}")
    print(f"Known labels: {known_mask.sum()} (SAFE={binary_labels.sum()}, UNSAFE={len(binary_labels)-binary_labels.sum()})")

    # Standardize
    scaler = StandardScaler()
    hidden_scaled = scaler.fit_transform(all_hidden)

    # Train probe
    probe = LogisticRegression(max_iter=1000, C=0.0001)
    probe.fit(hidden_scaled[known_mask], binary_labels)
    acc = probe.score(hidden_scaled[known_mask], binary_labels)
    print(f"Probe accuracy: {acc:.4f}")

    # Save
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"probe": probe, "scaler": scaler, "accuracy": acc}, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
