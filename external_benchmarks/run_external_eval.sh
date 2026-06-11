#!/bin/bash
# Run external benchmark evaluation with depth injection
#
# Usage:
#   ./external_benchmarks/run_external_eval.sh qwen3-4b
#   ./external_benchmarks/run_external_eval.sh --variant=full_interaction qwen3-4b
#   ./external_benchmarks/run_external_eval.sh --variant=full_interaction,compliant_response qwen3-4b
#   ./external_benchmarks/run_external_eval.sh --bench=asb qwen3-4b
#   ./external_benchmarks/run_external_eval.sh --steps=2 qwen3-4b
#
# Variants: full_interaction, compliant_response, random_response, empty_response

set -e
cd "$(dirname "$0")/.."

# Map human-readable variant names to internal IDs
variant_to_id() {
    case "$1" in
        full_interaction) echo "A1" ;;
        compliant_response) echo "A21" ;;
        random_response) echo "A22" ;;
        empty_response) echo "A23" ;;
        *) echo "$1" ;;  # pass through if already an ID
    esac
}

STEPS="1,2"
BENCH="asb,bfcl_multi,agentharm,api_bank"
VARIANT_STR="full_interaction,compliant_response,random_response,empty_response"
MODELS=()
DEPTHS="0,5,10,20"

for arg in "$@"; do
    if [[ "$arg" == --steps=* ]]; then
        STEPS="${arg#--steps=}"
    elif [[ "$arg" == --bench=* ]]; then
        BENCH="${arg#--bench=}"
    elif [[ "$arg" == --depths=* ]]; then
        DEPTHS="${arg#--depths=}"
    elif [[ "$arg" == --variant=* ]]; then
        VARIANT_STR="${arg#--variant=}"
    else
        MODELS+=("$arg")
    fi
done

[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("qwen3-4b")
IFS=',' read -ra RAW_VARIANTS <<< "$VARIANT_STR"
VARIANTS=()
for v in "${RAW_VARIANTS[@]}"; do
    VARIANTS+=("$(variant_to_id "$v")")
done

for MODEL in "${MODELS[@]}"; do
for VARIANT in "${VARIANTS[@]}"; do
    ASB_DIR="external_benchmarks/results/${MODEL}/${VARIANT}/asb"
    BFCL_MULTI_DIR="external_benchmarks/results/${MODEL}/${VARIANT}/bfcl_multi"

    echo "============================================================"
    echo "EXTERNAL EVAL: $MODEL | variant=$VARIANT | depths=$DEPTHS"
    echo "============================================================"

    # ASB Safety
    if [[ "$BENCH" == *asb* ]]; then
        mkdir -p "$ASB_DIR"

        # Step 1: Inference (skip if trajectories already exist)
        if [[ "$STEPS" == *1* ]]; then
            if [[ -f "$ASB_DIR/trajectories.json" ]]; then
                echo "=== ASB ($VARIANT): SKIP inference (trajectories.json exists) ==="
            else
                echo "=== ASB ($VARIANT): Run inference ==="
                python external_benchmarks/src/run_asb.py \
                    --model_path "$MODEL" \
                    --variant "$VARIANT" \
                    --depths "$DEPTHS" \
                    --output_dir "$ASB_DIR" \
                    --tensor_parallel_size 4 \
                    --n_runs 1
            fi
            echo ""
        fi

        # Step 2: Judge (skip if judgments already exist)
        if [[ "$STEPS" == *2* ]]; then
            if [[ -f "$ASB_DIR/judgments.json" ]]; then
                echo "=== ASB ($VARIANT): SKIP judge (judgments.json exists) ==="
            elif [[ -f "$ASB_DIR/trajectories.json" ]]; then
                echo "=== ASB ($VARIANT): Judge with ShieldAgent ==="
                python external_benchmarks/src/judge_asb_shield.py \
                    --input "$ASB_DIR/trajectories.json" \
                    --output "$ASB_DIR/judgments.json"
            fi
            echo ""
        fi
    fi

    # AgentHarm Safety
    if [[ "$BENCH" == *agentharm* ]]; then
        AGENTHARM_DIR="external_benchmarks/results/${MODEL}/${VARIANT}/agentharm"
        mkdir -p "$AGENTHARM_DIR"

        # Step 1: Inference (skip if trajectories exist)
        if [[ "$STEPS" == *1* ]]; then
            if [[ -f "$AGENTHARM_DIR/trajectories.json" ]]; then
                echo "=== AgentHarm ($VARIANT): SKIP inference (trajectories.json exists) ==="
            else
                echo "=== AgentHarm ($VARIANT): Run inference ==="
                python external_benchmarks/src/run_agentharm.py \
                    --model_path "$MODEL" \
                    --variant "$VARIANT" \
                    --depths "$DEPTHS" \
                    --output_dir "$AGENTHARM_DIR" \
                    --tensor_parallel_size 4 \
                    --n_runs 1
            fi
            echo ""
        fi

        # Step 2: Judge with SODA-Agent-Safety-Judge (skip if judgments exist)
        if [[ "$STEPS" == *2* ]]; then
            if [[ -f "$AGENTHARM_DIR/judgments.json" ]]; then
                echo "=== AgentHarm ($VARIANT): SKIP judge (judgments.json exists) ==="
            elif [[ -f "$AGENTHARM_DIR/trajectories.json" ]]; then
                echo "=== AgentHarm ($VARIANT): Judge with SODA-Agent-Safety-Judge ==="
                python external_benchmarks/src/judge_agentharm.py \
                    --input "$AGENTHARM_DIR/trajectories.json" \
                    --output "$AGENTHARM_DIR/judgments.json"
            fi
            echo ""
        fi
    fi

    # BFCL Single-Turn Utility

        # Step 1: Inference (skip if trajectories exist)
        if [[ "$STEPS" == *1* ]]; then
                echo "=== BFCL Single ($VARIANT): SKIP inference (trajectories.json exists) ==="
            else
                echo "=== BFCL Single ($VARIANT): Run inference ==="
                    --model_path "$MODEL" \
                    --variant "$VARIANT" \
                    --depths "$DEPTHS" \
                    --tensor_parallel_size 4 \
                    --n_runs 1
            fi
            echo ""
        fi

        # Step 2: Eval (skip if eval.json exists)
        if [[ "$STEPS" == *2* ]]; then
                echo "=== BFCL Single ($VARIANT): SKIP eval (eval.json exists) ==="
                echo "=== BFCL Single ($VARIANT): Evaluate ==="
            fi
            echo ""
        fi
    fi

    # BFCL Multi-Turn Utility
    if [[ "$BENCH" == *bfcl_multi* ]]; then
        mkdir -p "$BFCL_MULTI_DIR"

        # Step 1: Inference (skip if trajectories exist)
        if [[ "$STEPS" == *1* ]]; then
            if [[ -f "$BFCL_MULTI_DIR/trajectories.json" ]]; then
                echo "=== BFCL Multi ($VARIANT): SKIP inference (trajectories.json exists) ==="
            else
                echo "=== BFCL Multi ($VARIANT): Run inference ==="
                python external_benchmarks/src/run_bfcl.py \
                    --model_path "$MODEL" \
                    --variant "$VARIANT" \
                    --depths "$DEPTHS" \
                    --output_dir "$BFCL_MULTI_DIR" \
                    --tensor_parallel_size 4 \
                    --n_runs 1
            fi
            echo ""
        fi

        # Step 2: Eval (skip if eval.json exists)
        if [[ "$STEPS" == *2* ]]; then
            if [[ -f "$BFCL_MULTI_DIR/eval.json" ]]; then
                echo "=== BFCL Multi ($VARIANT): SKIP eval (eval.json exists) ==="
            elif [[ -f "$BFCL_MULTI_DIR/trajectories.json" ]]; then
                echo "=== BFCL Multi ($VARIANT): Evaluate ==="
                python external_benchmarks/src/eval_bfcl.py \
                    --input "$BFCL_MULTI_DIR/trajectories.json" \
                    --output "$BFCL_MULTI_DIR/eval.json"
            fi
            echo ""
        fi
    fi

    # API-Bank Multi-Turn Utility
    if [[ "$BENCH" == *api_bank* ]]; then
        API_BANK_DIR="external_benchmarks/results/${MODEL}/${VARIANT}/api_bank"
        mkdir -p "$API_BANK_DIR"

        # Step 1: Inference (skip if trajectories exist)
        if [[ "$STEPS" == *1* ]]; then
            if [[ -f "$API_BANK_DIR/trajectories.json" ]]; then
                echo "=== API-Bank ($VARIANT): SKIP inference (trajectories.json exists) ==="
            else
                echo "=== API-Bank ($VARIANT): Run inference ==="
                python external_benchmarks/src/run_api_bank.py \
                    --model_path "$MODEL" \
                    --variant "$VARIANT" \
                    --depths "$DEPTHS" \
                    --output_dir "$API_BANK_DIR" \
                    --tensor_parallel_size 4 \
                    --n_runs 1
            fi
            echo ""
        fi

        # Step 2: Eval (skip if eval.json exists)
        if [[ "$STEPS" == *2* ]]; then
            if [[ -f "$API_BANK_DIR/eval.json" ]]; then
                echo "=== API-Bank ($VARIANT): SKIP eval (eval.json exists) ==="
            elif [[ -f "$API_BANK_DIR/trajectories.json" ]]; then
                echo "=== API-Bank ($VARIANT): Evaluate ==="
                python external_benchmarks/src/eval_api_bank.py \
                    --input "$API_BANK_DIR/trajectories.json" \
                    --output "$API_BANK_DIR/eval.json"
            fi
            echo ""
        fi
    fi

    # Summary
    echo "=== Summary: $MODEL $VARIANT ==="
    if [[ -f "$ASB_DIR/judgments.json" ]]; then
        python3 -c "
import json
with open('$ASB_DIR/judgments.json') as f:
    data = json.load(f)
print(f'  ASB safety by depth:')
for d, s in sorted(data.get('by_depth', {}).items(), key=lambda x: int(x[0])):
    print(f'    d={d}: {s[\"mean\"]*100:.1f}% ± {s[\"std\"]*100:.1f}%')
"
    fi
        python3 -c "
import json
    data = json.load(f)
print(f'  BFCL single-turn by depth:')
for d, s in sorted(data.get('by_depth', {}).items(), key=lambda x: int(x[0])):
    print(f'    d={d}: {s[\"mean\"]*100:.1f}% ± {s[\"std\"]*100:.1f}%')
"
    fi
    if [[ -f "$BFCL_MULTI_DIR/eval.json" ]]; then
        python3 -c "
import json
with open('$BFCL_MULTI_DIR/eval.json') as f:
    data = json.load(f)
print(f'  BFCL multi-turn by depth:')
for d, s in sorted(data.get('by_depth', {}).items(), key=lambda x: int(x[0])):
    print(f'    d={d}: {s[\"mean\"]*100:.1f}% ± {s[\"std\"]*100:.1f}%')
"
    fi
    if [[ -f "${AGENTHARM_DIR:-/dev/null}/judgments.json" ]]; then
        python3 -c "
import json
with open('${AGENTHARM_DIR}/judgments.json') as f:
    data = json.load(f)
print(f'  AgentHarm safety by depth:')
for d, s in sorted(data.get('by_depth', {}).items(), key=lambda x: int(x[0])):
    print(f'    d={d}: {s[\"safety_rate\"]*100:.1f}% ({s[\"n_safe\"]}/{s[\"n_total\"]})')
"
    fi
    if [[ -f "${API_BANK_DIR:-/dev/null}/eval.json" ]]; then
        python3 -c "
import json
with open('${API_BANK_DIR}/eval.json') as f:
    data = json.load(f)
print(f'  API-Bank utility by depth:')
for d, s in sorted(data.get('by_depth', {}).items(), key=lambda x: int(x[0])):
    print(f'    d={d}: {s[\"param_accuracy\"]*100:.1f}% (name={s[\"name_accuracy\"]*100:.1f}%)')
"
    fi
    echo ""
done
done
