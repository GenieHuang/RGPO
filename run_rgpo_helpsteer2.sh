#!/bin/bash

source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2
export WANDB_PROJECT="rgpo_helpsteer2"

MODE="rgpo_dpo"  # dpo, rgpo_dpo, simpo, rgpo_simpo, ipo, rgpo_ipo
WEIGHT_SCALING="tanh"  # power, sigmoid, tanh, centralize, None
WEIGHT_SCALING_LAMBDA=1.0
BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"  # or Qwen/Qwen2.5-7B-Instruct
PREFERENCE_MODE="rgpo_predicted"  # uniform_ensemble, rgpo_predicted
ANNOTATION_DIM="helpfulness"
BETA=0.1
LEARNING_RATE=5e-6
SIMPO_GAMMA=1.375  # for SimPO and RGPO+SimPO only

if [ "$MODE" == "rgpo_dpo" ]; then
    USE_ROBUST="True"
    LOSS_TYPE="sigmoid"
    RUN_NAME="rgpo-dpo-helpsteer2-${ANNOTATION_DIM}-beta${BETA}-${BASE_MODEL}-$(date +%Y%m%d%H%M%S)"
    OUTPUT_DIR="./outputs/rgpo_dpo_helpsteer2_${ANNOTATION_DIM}_beta${BETA}_${BASE_MODEL}_$(date +%Y%m%d%H%M%S)"
elif [ "$MODE" == "simpo" ]; then
    USE_ROBUST="False"
    LOSS_TYPE="simpo"
    RUN_NAME="simpo-helpsteer2-${ANNOTATION_DIM}-beta${BETA}-gamma${SIMPO_GAMMA}-${BASE_MODEL}-$(date +%Y%m%d%H%M%S)"
    OUTPUT_DIR="./outputs/simpo_helpsteer2_${ANNOTATION_DIM}_beta${BETA}_gamma${SIMPO_GAMMA}_${BASE_MODEL}_$(date +%Y%m%d%H%M%S)"
elif [ "$MODE" == "rgpo_simpo" ]; then
    USE_ROBUST="True"
    LOSS_TYPE="simpo"
    RUN_NAME="rgpo-simpo-helpsteer2-${ANNOTATION_DIM}-beta${BETA}-gamma${SIMPO_GAMMA}-${BASE_MODEL}-$(date +%Y%m%d%H%M%S)"
    OUTPUT_DIR="./outputs/rgpo_simpo_helpsteer2_${ANNOTATION_DIM}_beta${BETA}_gamma${SIMPO_GAMMA}_${BASE_MODEL}_$(date +%Y%m%d%H%M%S)"
elif [ "$MODE" == "ipo" ]; then
    USE_ROBUST="False"
    LOSS_TYPE="ipo"
    RUN_NAME="ipo-helpsteer2-${ANNOTATION_DIM}-beta${BETA}-${BASE_MODEL}-$(date +%Y%m%d%H%M%S)"
    OUTPUT_DIR="./outputs/ipo_helpsteer2_${ANNOTATION_DIM}_beta${BETA}_${BASE_MODEL}_$(date +%Y%m%d%H%M%S)"
elif [ "$MODE" == "rgpo_ipo" ]; then
    USE_ROBUST="True"
    LOSS_TYPE="ipo"
    RUN_NAME="rgpo-ipo-helpsteer2-${ANNOTATION_DIM}-beta${BETA}-${BASE_MODEL}-$(date +%Y%m%d%H%M%S)"
    OUTPUT_DIR="./outputs/rgpo_ipo_helpsteer2_${ANNOTATION_DIM}_beta${BETA}_${BASE_MODEL}_$(date +%Y%m%d%H%M%S)"
else
    USE_ROBUST="False"
    LOSS_TYPE="sigmoid"
    RUN_NAME="dpo-dpo-helpsteer2-${ANNOTATION_DIM}-beta${BETA}-${BASE_MODEL}-$(date +%Y%m%d%H%M%S)"
    OUTPUT_DIR="./outputs/dpo_dpo_helpsteer2_${ANNOTATION_DIM}_beta${BETA}_${BASE_MODEL}_$(date +%Y%m%d%H%M%S)"
fi

deepspeed --include localhost:0,1,2 train_rgpo.py \
    --deepspeed ds_config_zero2.json \
    --model_name "${BASE_MODEL}" \
    --train_data_path "data/train/helpsteer2_disagreement_paired.json" \
    --dawid_skene_results_path "maximum_like_est/estimated_helpsteer2_${ANNOTATION_DIM}_with_ties.json" \
    --annotation_dim "${ANNOTATION_DIM}" \
    --num_annotators 0 \
    --threshold 0.0 \
    --weight_scaling "${WEIGHT_SCALING}" \
    --weight_scaling_lambda ${WEIGHT_SCALING_LAMBDA} \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --wandb_run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type "constant" \
    --weight_decay 0.01 \
    --bf16 True \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --save_total_limit 8 \
    --seed 42 \
    --beta ${BETA} \
    --max_length 2048 \
    --max_prompt_length 1024 \
    --gradient_checkpointing True \
    --use_consistency_weighted ${USE_ROBUST} \
    --preference_mode "${PREFERENCE_MODE}" \
    --loss_type "${LOSS_TYPE}" \
    --simpo_gamma ${SIMPO_GAMMA}
