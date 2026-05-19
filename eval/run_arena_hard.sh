#!/bin/bash

#   ./arena_hard_eval.sh                    # Evaluate all models
#   ./arena_hard_eval.sh --generate-only    # Only generate answers (skip judgment)
#   ./arena_hard_eval.sh --judge-only       # Only run judgment (skip answer generation)

set -e

GENERATE_ONLY=false
JUDGE_ONLY=false
for arg in "$@"; do
    case $arg in
        --generate-only)
            GENERATE_ONLY=true
            shift
            ;;
        --judge-only)
            JUDGE_ONLY=true
            shift
            ;;
    esac
done

if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

if [ -z "$OPENAI_API_KEY" ] && [ "$GENERATE_ONLY" = false ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "Please add it to your .env file:"
    echo "  OPENAI_API_KEY=your_api_key"
    exit 1
fi

GPU_IDS=${GPU_IDS:-"0,1"}
VLLM_PORT=${VLLM_PORT:-8000}
ARENA_HARD_DIR="arena-hard-auto"
JUDGE_MODEL="gpt-4-1106-preview"
BENCH_NAME="arena-hard-v0.1"
REFERENCE_MODEL="gpt-4-0314"

# List of models to evaluate
MODELS=(
    # "outputs/rgpo_dpo_multipref_helpful_beta0.05_meta-llama/Meta-Llama-3-8B-Instruct"
)

get_model_name() {
    local model_path=$1
    if [[ "$model_path" == "meta-llama/Meta-Llama-3-8B-Instruct" ]]; then
        echo "llama3-8b-instruct"
    elif [[ "$model_path" == "Qwen/Qwen2.5-7B-Instruct" ]]; then
        echo "qwen2.5-7b-instruct"
    else
        local dir_name=$(dirname "$model_path" | sed 's|outputs/||')
        local base_name=$(basename "$model_path")
        local timestamp=$(echo "$base_name" | grep -oE '[0-9]{14}$')
        local short_name=$(echo "${dir_name}" | sed 's|/|_|g')
        if [ -n "$timestamp" ]; then
            echo "${short_name}_${timestamp}"
        else
            echo "${short_name}"
        fi
    fi
}

is_lora_adapter() {
    local model_path=$1
    if [ -f "${model_path}/adapter_config.json" ]; then
        return 0  # true
    else
        return 1  # false
    fi
}

get_model_path_for_vllm() {
    local model_path=$1
    if is_lora_adapter "${model_path}"; then
        # Use merged version
        echo "${model_path}_merged"
    else
        echo "${model_path}"
    fi
}

start_vllm_server() {
    local model_path=$1
    local actual_model_path=$(get_model_path_for_vllm "${model_path}")

    echo "Starting vLLM server for: ${model_path}"
    echo "Using GPUs: ${GPU_IDS}"

    pkill -f "vllm.entrypoints.openai.api_server.*${VLLM_PORT}" 2>/dev/null || true
    sleep 2

    if is_lora_adapter "${model_path}"; then
        echo "Detected LoRA adapter, using merged model: ${actual_model_path}"

        # Check if merged model exists
        if [ ! -d "${actual_model_path}" ] || [ ! -f "${actual_model_path}/config.json" ]; then
            echo "ERROR: Merged model not found at ${actual_model_path}"
            echo "Please run the merge script first:"
            echo "  python eval/merge_lora_adapters.py --adapter-path ${model_path}"
            echo "Or merge all adapters at once:"
            echo "  python eval/merge_lora_adapters.py --adapters-file eval/adapters_to_merge.txt"
            return 1
        fi
    fi

    NUM_GPUS=$(echo ${GPU_IDS} | tr ',' '\n' | wc -l)

    CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m vllm.entrypoints.openai.api_server \
        --model ${actual_model_path} \
        --port ${VLLM_PORT} \
        --trust-remote-code \
        --max-model-len 8192 \
        --tensor-parallel-size ${NUM_GPUS} \
        > /tmp/vllm_server.log 2>&1 &

    VLLM_PID=$!
    echo "vLLM server started with PID: ${VLLM_PID}"

    echo "Waiting for vLLM server to be ready..."
    for i in {1..120}; do
        if curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            return 0
        fi
        if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            return 0
        fi
        if [ $((i % 6)) -eq 0 ]; then
            echo "  Still waiting... (${i}0 seconds elapsed)"
            if ! kill -0 ${VLLM_PID} 2>/dev/null; then
                echo "ERROR: vLLM process died!"
                echo "=== vLLM server log ==="
                cat /tmp/vllm_server.log
                return 1
            fi
        fi
        sleep 5
    done

    echo "ERROR: vLLM server failed to start within 10 minutes"
    echo "=== vLLM server log ==="
    cat /tmp/vllm_server.log
    return 1
}

stop_vllm_server() {
    echo "Stopping vLLM server..."
    pkill -f "vllm.entrypoints.openai.api_server.*${VLLM_PORT}" 2>/dev/null || true
    sleep 2
}

update_arena_hard_config() {
    local model_name=$1
    local model_path=$2
    local actual_model_path=$(get_model_path_for_vllm "${model_path}")

    cat > ${ARENA_HARD_DIR}/config/api_config_local.yaml << EOF
${model_name}:
    model: ${actual_model_path}
    endpoints:
        - api_base: http://localhost:${VLLM_PORT}/v1
          api_key: '-'
    api_type: openai
    parallel: 32
    max_tokens: 4096
    temperature: 0.0

${JUDGE_MODEL}:
    model: gpt-4-1106-preview
    endpoints: null
    api_type: openai
    parallel: 128
    max_tokens: 8196
    temperature: 0.0
EOF

    # Update gen_answer_config.yaml
    cat > ${ARENA_HARD_DIR}/config/gen_answer_config.yaml << EOF
bench_name: ${BENCH_NAME}
model_list:
  - ${model_name}
EOF

    # Update judgment config yaml
    cat > ${ARENA_HARD_DIR}/config/${BENCH_NAME}.yaml << EOF
judge_model: ${JUDGE_MODEL}
temperature: 0.0
max_tokens: 8196

bench_name: ${BENCH_NAME}

reference:
  - ${REFERENCE_MODEL}

regex_patterns:
  - \\[\\[([AB<>=]+)\\]\\]
  - \\[([AB<>=]+)\\]

prompt_template: "<|User Prompt|>\n{QUESTION}\n\n<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n<|The End of Assistant B's Answer|>"

model_list:
  - ${model_name}
EOF
}

generate_answers() {
    local model_name=$1

    echo "Generating answers for: ${model_name}"
    cd ${ARENA_HARD_DIR}
    python gen_answer.py --config-file config/gen_answer_config.yaml --endpoint-file config/api_config_local.yaml
    cd ..
}

generate_judgments() {
    local model_name=$1

    echo "Generating judgments for: ${model_name}"
    cd ${ARENA_HARD_DIR}
    python gen_judgment.py --setting-file config/${BENCH_NAME}.yaml --endpoint-file config/api_config_local.yaml
    cd ..
}

echo "=========================================="
echo "Benchmark:   ${BENCH_NAME}"
echo "Reference:   ${REFERENCE_MODEL}"
echo "Judge Model: ${JUDGE_MODEL}"
echo "GPUs:        ${GPU_IDS} (tensor parallel)"
echo "vLLM Port:   ${VLLM_PORT}"
echo "Models:      ${#MODELS[@]}"
echo ""

# Create results tracking file
RESULTS_FILE="arena_hard_results_$(date +%Y%m%d_%H%M%S).txt"
echo "Arena-Hard-Auto Results" > ${RESULTS_FILE}
echo "Benchmark: ${BENCH_NAME}" >> ${RESULTS_FILE}
echo "Reference: ${REFERENCE_MODEL}" >> ${RESULTS_FILE}
echo "Judge: ${JUDGE_MODEL}" >> ${RESULTS_FILE}
echo "Date: $(date)" >> ${RESULTS_FILE}
echo "==========================================" >> ${RESULTS_FILE}
echo "" >> ${RESULTS_FILE}


for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    MODEL_NAME=$(get_model_name "${MODEL_PATH}")
    MODEL_NUM=$((i + 1))

    echo ""
    echo "=========================================="
    echo "[${MODEL_NUM}/${#MODELS[@]}] Processing: ${MODEL_NAME}"
    echo "Path: ${MODEL_PATH}"
    echo "=========================================="

    update_arena_hard_config "${MODEL_NAME}" "${MODEL_PATH}"

    if [ "$JUDGE_ONLY" = false ]; then
        start_vllm_server "${MODEL_PATH}"

        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to start vLLM server for ${MODEL_NAME}"
            echo "${MODEL_NAME}: FAILED (vLLM server)" >> ${RESULTS_FILE}
            continue
        fi

        generate_answers "${MODEL_NAME}"
        stop_vllm_server
    fi

    if [ "$GENERATE_ONLY" = false ]; then
        generate_judgments "${MODEL_NAME}"
    fi

    echo ""
    echo "[${MODEL_NUM}/${#MODELS[@]}] Completed: ${MODEL_NAME}"
    echo "${MODEL_NAME}: COMPLETED" >> ${RESULTS_FILE}
done

echo ""
echo "====================Done======================"

if [ "$GENERATE_ONLY" = false ]; then
    # Show final results
    echo ""
    echo "Final Results (vs ${REFERENCE_MODEL}):"
    echo "=========================================="
    cd ${ARENA_HARD_DIR}
    python show_result.py --benchmark ${BENCH_NAME} --judge-names ${JUDGE_MODEL} --category ${BENCH_NAME}
    cd ..
fi

echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo ""
