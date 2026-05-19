#!/bin/bash

#   ./compare_two_models_alpaca_eval.sh           # Full evaluation (805 samples)
#   ./compare_two_models_alpaca_eval.sh 10        # Quick test (10 samples)
#   ./compare_two_models_alpaca_eval.sh 100       # Medium test (100 samples)

set -e

NUM_EXAMPLES=${1:-}

if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

if [ -d "venv" ]; then
    source venv/bin/activate
fi

export HF_DATASETS_TRUST_REMOTE_CODE=1

GPU_ID_1=0
GPU_ID_2=1

BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
ANNOTATOR="weighted_alpaca_eval_gpt4_turbo"

# e.g. MODEL_PATH="outputs/rgpo_dpo_multipref_helpful_beta0.05_meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_PATH=""
MODEL_DIR=$(dirname "$MODEL_PATH" | sed 's|outputs/||')
MODEL_BASE=$(basename "$MODEL_PATH")
MODEL_SHORT=$(echo "$MODEL_DIR" | sed 's/robust_dpo_helpsteer2_/robust_/' | sed 's/_meta-llama$//')
OUTPUT_DIR="alpaca_eval_results/${MODEL_SHORT}_${MODEL_BASE}"

if [ -z "$NUM_EXAMPLES" ]; then
    MODEL_NAME="robust_dpo_helpsteer2_helpfulness_beta0.1"
    NUM_EXAMPLES_TEXT="805 (full)"
    NUM_EXAMPLES_FLAG=""
else
    MODEL_NAME="robust_dpo_helpsteer2_helpfulness_beta0.1_test${NUM_EXAMPLES}"
    NUM_EXAMPLES_TEXT="${NUM_EXAMPLES} (test mode)"
    NUM_EXAMPLES_FLAG="--num_examples ${NUM_EXAMPLES}"
fi

mkdir -p ${OUTPUT_DIR}

echo ""
echo "Model:        ${MODEL_PATH}"
echo "GPUs:         ${GPU_ID_1}, ${GPU_ID_2} (parallel processing)"
echo "Output Dir:   ${OUTPUT_DIR}"
echo "Samples:      ${NUM_EXAMPLES_TEXT}"
echo ""

# Generate outputs
echo "=========================================="
if [ -z "$NUM_EXAMPLES" ]; then
    TOTAL_EXAMPLES=805
else
    TOTAL_EXAMPLES=${NUM_EXAMPLES}
fi

MIDPOINT=$((TOTAL_EXAMPLES / 2))
echo "Processing ${TOTAL_EXAMPLES} examples:"
echo "  GPU ${GPU_ID_1}: examples 0-${MIDPOINT}"
echo "  GPU ${GPU_ID_2}: examples ${MIDPOINT}-${TOTAL_EXAMPLES}"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID_1} python eval/alpaca_eval_inference.py \
    --adapter_path ${MODEL_PATH} \
    --base_model ${BASE_MODEL} \
    --output_path ${OUTPUT_DIR}/${MODEL_NAME}_outputs_gpu${GPU_ID_1}.json \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --start_index 0 \
    --end_index ${MIDPOINT} \
    ${NUM_EXAMPLES_FLAG} &

PID_GPU1=$!

CUDA_VISIBLE_DEVICES=${GPU_ID_2} python eval/alpaca_eval_inference.py \
    --adapter_path ${MODEL_PATH} \
    --base_model ${BASE_MODEL} \
    --output_path ${OUTPUT_DIR}/${MODEL_NAME}_outputs_gpu${GPU_ID_2}.json \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --start_index ${MIDPOINT} \
    --end_index ${TOTAL_EXAMPLES} \
    ${NUM_EXAMPLES_FLAG} &

PID_GPU2=$!

echo "GPU ${GPU_ID_1} process: PID ${PID_GPU1}"
echo "GPU ${GPU_ID_2} process: PID ${PID_GPU2}"
echo ""

wait ${PID_GPU1}
EXIT_CODE_1=$?

wait ${PID_GPU2}
EXIT_CODE_2=$?

if [ ${EXIT_CODE_1} -ne 0 ] || [ ${EXIT_CODE_2} -ne 0 ]; then
    echo ""
    echo "ERROR: One or both inference processes failed!"
    echo "  GPU ${GPU_ID_1} exit code: ${EXIT_CODE_1}"
    echo "  GPU ${GPU_ID_2} exit code: ${EXIT_CODE_2}"
    exit 1
fi

echo ""
echo "====================Done======================"
echo ""

python3 -c "
import json

with open('${OUTPUT_DIR}/${MODEL_NAME}_outputs_gpu${GPU_ID_1}.json', 'r') as f:
    outputs_gpu1 = json.load(f)

with open('${OUTPUT_DIR}/${MODEL_NAME}_outputs_gpu${GPU_ID_2}.json', 'r') as f:
    outputs_gpu2 = json.load(f)

merged_outputs = outputs_gpu1 + outputs_gpu2

with open('${OUTPUT_DIR}/${MODEL_NAME}_outputs.json', 'w') as f:
    json.dump(merged_outputs, f, indent=2, ensure_ascii=False)

print(f'Merged {len(outputs_gpu1)} + {len(outputs_gpu2)} = {len(merged_outputs)} outputs')
"

echo ""
echo "Final output: ${OUTPUT_DIR}/${MODEL_NAME}_outputs.json"
echo ""

# Run AlpacaEval evaluation
echo "=========================================="

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "Please add it to your .env file:"
    echo "  OPENAI_API_KEY=your_api_key"
    exit 1
fi

echo "OPENAI_API_KEY found in environment ✓"
echo ""

# Run evaluation
alpaca_eval --model_outputs ${OUTPUT_DIR}/${MODEL_NAME}_outputs.json \
    --reference_outputs data/alpaca_eval/alpaca_eval_gpt4_baseline.json \
    --annotators_config ${ANNOTATOR} \
    --output_path ${OUTPUT_DIR}/evaluation_results

echo ""
echo "====================Done======================"
echo ""
echo "View results:"
echo "  cat ${OUTPUT_DIR}/evaluation_results/*/leaderboard.csv"
echo ""
