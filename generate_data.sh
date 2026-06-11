#!/usr/bin/env bash
set -e

# ==============================================================================
# Generate SODA benchmark data locally (optional)
#
# By default, evaluation scripts auto-download data from HuggingFace.
# Run this script only if you want to generate/regenerate data locally.
#
# Note: random_request and empty_request variants require full_interaction
# trajectories. Run the full_interaction evaluation first, then use
# --trajectories to point to the generated trajectories.json.
#
# Usage:
#   bash generate_data.sh                                  # Generate all standalone variants
#   bash generate_data.sh --with-request-variants MODEL    # Also generate request variants
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p data

echo "=== Generating full_interaction tasks ==="
python src/generate_tasks.py

echo ""
echo "=== Generating standalone ablation variants ==="
for variant in compliant_response random_response empty_response all_random all_empty; do
    echo "  Generating $variant..."
    python src/generate_tasks_ablation.py \
        --variant "$variant" \
        --output "data/tasks_${variant}.json"
done

# Request-side variants need full_interaction trajectories
if [[ "$1" == "--with-request-variants" && -n "$2" ]]; then
    MODEL="$2"
    TRAJ="results/${MODEL}/full_interaction/trajectories.json"
    if [[ ! -f "$TRAJ" ]]; then
        echo ""
        echo "ERROR: $TRAJ not found."
        echo "Run full_interaction evaluation first: ./run_pipeline.sh --steps=1 $MODEL"
        exit 1
    fi
    echo ""
    echo "=== Generating request-side variants (using $MODEL trajectories) ==="
    for variant in random_request empty_request; do
        echo "  Generating $variant..."
        python src/generate_tasks_ablation.py \
            --variant "$variant" \
            --output "data/tasks_${variant}.json" \
            --trajectories "$TRAJ"
    done
fi

echo ""
echo "=== Done ==="
ls -lh data/tasks_*.json
