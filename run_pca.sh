#!/usr/bin/env bash
set -e

# ==============================================================================
# Representation Analysis Pipeline
#
# Extracts hidden states, trains safety probe, and generates PCA figures.
#
# Usage:
#   ./run_pca.sh qwen3-4b                  # Single model
#   ./run_pca.sh qwen3-4b llama3.1-8b      # Multiple models
#   ./run_pca.sh --steps=3 qwen3-4b        # Plot only (if data exists)
#
# Steps:
#   1. Extract hidden states from full_interaction trajectories
#   2. Train safety probe (logistic regression on hidden states)
#   3. Plot PCA visualization
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STEPS="1,2,3"
MODELS=()

for arg in "$@"; do
    if [[ "$arg" == --steps=* ]]; then
        STEPS="${arg#--steps=}"
    else
        MODELS+=("$arg")
    fi
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=("qwen3-4b")
fi

for MODEL in "${MODELS[@]}"; do
    echo "============================================================"
    echo "  Representation Analysis: $MODEL"
    echo "============================================================"

    # Step 1: Extract hidden states
    if [[ "$STEPS" == *1* ]]; then
        echo "[Step 1/3] Extracting hidden states..."
        python src/extract_hidden_states.py \
            --model_path "$MODEL" \
            --variant full_interaction \
            --output_dir hidden_states \
            --tensor_parallel_size 4
    fi

    # Step 2: Train safety probe
    if [[ "$STEPS" == *2* ]]; then
        echo "[Step 2/3] Training safety probe..."
        python src/train_safety_probe.py \
            --model "$MODEL" \
            --hidden_dir hidden_states \
            --results_dir results
    fi

    # Step 3: Plot PCA
    if [[ "$STEPS" == *3* ]]; then
        echo "[Step 3/3] Plotting PCA..."
        python src/plot_pca.py \
            --models "$MODEL" \
            --output "figures/pca_${MODEL}.pdf"
    fi

    echo ""
done

echo "Done."
