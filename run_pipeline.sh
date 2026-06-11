#!/usr/bin/env bash
set -e

# ==============================================================================
# SODA Evaluation Pipeline
#
# Runs the cold-start safety evaluation. Data auto-downloads from HuggingFace
# if not found locally. To generate data locally, run generate_data.sh first.
#
# Steps:
#   1. Run multi-turn agent inference
#   2. Judge trajectories with SODA-Agent-Safety-Judge
#   3. Pack results (aggregate safety rates by depth)
#
# Usage:
#   ./run_pipeline.sh qwen3-4b                           # Full Interaction (default)
#   ./run_pipeline.sh --variant compliant_response qwen3-4b
#   ./run_pipeline.sh --variant all qwen3-4b             # All 8 variants
#   ./run_pipeline.sh --steps 2 qwen3-4b                 # Judge only
#
# Variants:
#   full_interaction    Real agentic interaction (default, Table 1)
#   compliant_response  Real requests + "Sure, I can help" response
#   random_response     Real requests + random text response
#   empty_response      Real requests + empty response
#   random_request      Random text request + real response
#   empty_request       Empty request + real response
#   all_random          Both sides random text
#   all_empty           Both sides empty (chat template only)
#   all                 Run all 8 variants
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Variant name to task file mapping
declare -A VARIANT_TO_FILE=(
    ["full_interaction"]="data/tasks_full_interaction.json"
    ["compliant_response"]="data/tasks_compliant_response.json"
    ["random_response"]="data/tasks_random_response.json"
    ["empty_response"]="data/tasks_empty_response.json"
    ["random_request"]="data/tasks_random_request.json"
    ["empty_request"]="data/tasks_empty_request.json"
    ["all_random"]="data/tasks_all_random.json"
    ["all_empty"]="data/tasks_all_empty.json"
)

ALL_VARIANTS=(full_interaction compliant_response random_response empty_response random_request empty_request all_random all_empty)

# Parse arguments
STEPS="1,2,3"
VARIANT="full_interaction"
N_RUNS=3
MODELS=()

for arg in "$@"; do
    if [[ "$arg" == --steps=* ]]; then
        STEPS="${arg#--steps=}"
    elif [[ "$arg" == --variant=* ]]; then
        VARIANT="${arg#--variant=}"
    elif [[ "$arg" == --runs=* ]]; then
        N_RUNS="${arg#--runs=}"
    else
        MODELS+=("$arg")
    fi
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=("qwen3-4b")
fi

# Determine which variants to run
if [[ "$VARIANT" == "all" ]]; then
    VARIANTS=("${ALL_VARIANTS[@]}")
else
    VARIANTS=("$VARIANT")
fi

# Determine if variant needs prefilled eval
needs_prefill() {
    local v="$1"
    [[ "$v" != "full_interaction" ]]
}

for MODEL in "${MODELS[@]}"; do
    for V in "${VARIANTS[@]}"; do
        TASK_FILE="${VARIANT_TO_FILE[$V]}"
        OUT_DIR="results/${MODEL}/${V}"

        echo "============================================================"
        echo "  Model: $MODEL | Variant: $V"
        echo "  Tasks: $TASK_FILE | Output: $OUT_DIR"
        echo "============================================================"

        mkdir -p "$OUT_DIR"

        # Step 1: Run inference (skip if trajectories exist)
        if [[ "$STEPS" == *1* ]]; then
            if [[ -f "$OUT_DIR/trajectories.json" ]]; then
                echo "[Step 1] SKIP inference (trajectories.json exists)"
            else
                echo "[Step 1] Running inference ($N_RUNS runs)..."
                if needs_prefill "$V"; then
                    python src/run_eval_prefilled.py \
                        --model_path "$MODEL" \
                        --tasks "$TASK_FILE" \
                        --output_dir "$OUT_DIR" \
                        --tensor_parallel_size 4 \
                        --n_runs "$N_RUNS"
                else
                    python src/run_eval.py \
                        --model_path "$MODEL" \
                        --tasks "$TASK_FILE" \
                        --output_dir "$OUT_DIR" \
                        --tensor_parallel_size 4 \
                        --n_runs "$N_RUNS"
                fi
            fi
        fi

        # Step 2: Judge trajectories (skip if judgments exist)
        if [[ "$STEPS" == *2* ]]; then
            if [[ -f "$OUT_DIR/judgments.json" ]]; then
                echo "[Step 2] SKIP judge (judgments.json exists)"
            elif [[ -f "$OUT_DIR/trajectories.json" ]]; then
                echo "[Step 2] Judging trajectories..."
                python src/judge.py \
                    --input "$OUT_DIR/trajectories.json" \
                    --output "$OUT_DIR/judgments.json"
            else
                echo "[Step 2] SKIP judge (no trajectories.json)"
            fi
        fi

        # Step 3: Pack results (skip if stats.json exists)
        if [[ "$STEPS" == *3* ]]; then
            if [[ -f "$OUT_DIR/stats.json" ]]; then
                echo "[Step 3] SKIP pack (stats.json exists)"
            elif [[ -f "$OUT_DIR/judgments.json" ]]; then
                echo "[Step 3] Packing results..."
                python src/pack_results.py --eval_dir "$OUT_DIR"
            else
                echo "[Step 3] SKIP pack (no judgments.json)"
            fi
        fi

        echo ""
    done
done

echo "Pipeline complete."
