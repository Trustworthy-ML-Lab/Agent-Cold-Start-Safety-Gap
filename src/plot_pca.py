#!/usr/bin/env python3
"""Plot PCA visualization of hidden states colored by safety verdict.

Shows how representations migrate from unsafe to safe region as depth increases.

Usage:
    # Single model
    python src/plot_pca.py --models qwen3-4b --output figures/pca_qwen3-4b.pdf

    # Multiple models (paper-style grid)
    python src/plot_pca.py --models qwen3-4b llama3.1-8b gemma4-4b --output figures/pca_all.pdf
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
HIDDEN_DIR = ROOT / "hidden_states"
RESULTS_DIR = ROOT / "results"

VARIANT = "full_interaction"
DEPTHS_TO_PLOT = [0, 1, 3, 5, 10]

COLOR_SAFE = "#2979FF"
COLOR_UNSAFE = "#E53935"
COLOR_BOUNDARY = "#444444"

MODEL_LABELS = {
    "llama3.1-8b": "Llama-3.1-8B",
    "llama3.3-70b": "Llama-3.3-70B",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-30b-moe": "Qwen3-30B",
    "qwen3.5-9b": "Qwen3.5-9B",
    "gemma4-4b": "Gemma4-4B",
    "gemma4-26b-moe": "Gemma4-26B",
}


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


def plot_models(models, output):
    n_models = len(models)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    })

    fig, axes = plt.subplots(
        n_models, len(DEPTHS_TO_PLOT),
        figsize=(13, 2.5 * n_models),
        gridspec_kw={"hspace": 0.2, "wspace": 0.1},
        squeeze=False,
    )

    for row_idx, model_name in enumerate(models):
        model_label = MODEL_LABELS.get(model_name, model_name)

        hidden_path = HIDDEN_DIR / f"{model_name}_{VARIANT}_v2.npz"
        # Also try without _v2 suffix
        if not hidden_path.exists():
            hidden_path = HIDDEN_DIR / f"{model_name}_{VARIANT}.npz"
        if not hidden_path.exists():
            print(f"  Skipping {model_name}: {hidden_path} not found")
            for col_idx in range(len(DEPTHS_TO_PLOT)):
                axes[row_idx, col_idx].set_visible(False)
            continue

        judgments_path = RESULTS_DIR / model_name / VARIANT / "judgments.json"
        if not judgments_path.exists():
            print(f"  Skipping {model_name}: {judgments_path} not found")
            for col_idx in range(len(DEPTHS_TO_PLOT)):
                axes[row_idx, col_idx].set_visible(False)
            continue

        hidden, meta = load_hidden_states(str(hidden_path))
        verdicts = load_judgments(str(judgments_path))

        indices = [i for i, m in enumerate(meta) if m.get("is_threat", True)]
        hidden_filtered = hidden[indices]
        meta_filtered = [meta[i] for i in indices]

        labels = []
        depths = []
        for m in meta_filtered:
            task_id = m["id"]
            verdict = verdicts.get(task_id, "UNKNOWN")
            labels.append(verdict)
            depths.append(m["depth"])

        labels = np.array(labels)
        depths = np.array(depths)

        known_mask = labels != "UNKNOWN"
        hidden_known = hidden_filtered[known_mask]
        depths_known = depths[known_mask]
        labels_known = labels[known_mask]

        scaler = StandardScaler()
        hidden_scaled = scaler.fit_transform(hidden_known)
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(hidden_scaled)

        x_center = (embedding[:, 0].min() + embedding[:, 0].max()) / 2
        y_center = (embedding[:, 1].min() + embedding[:, 1].max()) / 2
        half_range = max(
            embedding[:, 0].max() - embedding[:, 0].min(),
            embedding[:, 1].max() - embedding[:, 1].min(),
        ) / 2 + 0.5
        x_min, x_max = x_center - half_range, x_center + half_range
        y_min, y_max = y_center - half_range, y_center + half_range

        # Load probe for decision boundary
        probe_path = HIDDEN_DIR / f"probe_{model_name}.pkl"
        boundary_data = None
        if probe_path.exists():
            with open(probe_path, "rb") as f:
                probe_data = pickle.load(f)
            probe = probe_data["probe"]
            probe_scaler = probe_data["scaler"]

            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 200),
                np.linspace(y_min, y_max, 200),
            )
            grid_pca = np.c_[xx.ravel(), yy.ravel()]
            grid_original = pca.inverse_transform(grid_pca)
            grid_unscaled = scaler.inverse_transform(grid_original)
            grid_probe_scaled = probe_scaler.transform(grid_unscaled)
            grid_probs = probe.predict_proba(grid_probe_scaled)[:, 1].reshape(xx.shape)
            boundary_data = (xx, yy, grid_probs)

        for col_idx, d in enumerate(DEPTHS_TO_PLOT):
            ax = axes[row_idx, col_idx]
            mask_d = depths_known == d
            emb_d = embedding[mask_d]
            lab_d = labels_known[mask_d]

            if boundary_data is not None:
                bxx, byy, bprobs = boundary_data
                ax.contourf(
                    bxx, byy, bprobs, levels=[0, 0.5, 1],
                    colors=["#FFCDD2", "#BBDEFB"], alpha=0.15,
                )
                ax.contour(
                    bxx, byy, bprobs, levels=[0.5],
                    colors=[COLOR_BOUNDARY], linewidths=1.0, linestyles="--",
                )

            unsafe_mask = lab_d == "UNSAFE"
            safe_mask = lab_d == "SAFE"

            ax.scatter(
                emb_d[unsafe_mask, 0], emb_d[unsafe_mask, 1],
                c=COLOR_UNSAFE, s=6, alpha=0.5, edgecolors="none", zorder=2,
            )
            ax.scatter(
                emb_d[safe_mask, 0], emb_d[safe_mask, 1],
                c=COLOR_SAFE, s=6, alpha=0.5, edgecolors="none", zorder=3,
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")

            if row_idx == 0:
                ax.set_title(f"$D = {d}$", fontsize=12, fontweight="bold", pad=6)

            if col_idx == 0:
                ax.text(
                    -0.15, 0.5, model_label,
                    transform=ax.transAxes, fontsize=12, fontweight="bold",
                    ha="right", va="center", rotation=0,
                )

    legend_elements = [
        Patch(facecolor=COLOR_SAFE, edgecolor="none", label="Safe"),
        Patch(facecolor=COLOR_UNSAFE, edgecolor="none", label="Unsafe"),
        plt.Line2D([0], [0], color=COLOR_BOUNDARY, linestyle="--", linewidth=1.2,
                   label="Estimated boundary"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=3, fontsize=12, frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")

    if output.endswith(".pdf"):
        png_path = output.replace(".pdf", ".png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {png_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot PCA visualization of hidden states.")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names (matching hidden_states/ and results/ directories)")
    parser.add_argument("--output", default="figures/pca.pdf",
                        help="Output file path (PDF or PNG)")
    parser.add_argument("--depths", type=str, default=None,
                        help="Comma-separated depths to plot (default: 0,1,3,5,10)")
    args = parser.parse_args()

    if args.depths:
        global DEPTHS_TO_PLOT
        DEPTHS_TO_PLOT = [int(d) for d in args.depths.split(",")]

    plot_models(args.models, args.output)


if __name__ == "__main__":
    main()
