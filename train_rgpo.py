"""
RGPO (Reliability-Guided Preference Optimization) training script.
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed

from trl import DPOConfig, DPOTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_weight_scaling(weights: np.ndarray, scaling_type: str, lmbda: float = 1.0) -> np.ndarray:
    """Apply weight scaling: power, sigmoid, tanh, or centralize."""
    if scaling_type is None or scaling_type.lower() == 'none':
        return weights

    if scaling_type == 'power':
        exponent = lmbda  # use provided lambda

        weights_pow = np.power(weights, exponent)
        # Normalize to [0, 1]
        min_w = weights_pow.min()
        max_w = weights_pow.max()
        if max_w - min_w > 1e-9:
            weights_norm = (weights_pow - min_w) / (max_w - min_w)
        else:
            weights_norm = weights_pow

        logger.info(f"Applied power scaling (λ={exponent}): mean={np.mean(weights_norm):.4f}, std={np.std(weights_norm):.4f}")
        return weights_norm

    elif scaling_type == 'sigmoid':
        scale = lmbda  # use lmbda as scale

        mean_w = np.mean(weights)
        std_w = np.std(weights)

        if std_w > 1e-9:
            z_scores = (weights - mean_w) / std_w
        else:
            z_scores = weights - mean_w

        weights_sigmoid = 1 / (1 + np.exp(-z_scores * scale))

        logger.info(f"Applied sigmoid scaling (scale={scale}): mean={np.mean(weights_sigmoid):.4f}, std={np.std(weights_sigmoid):.4f}")
        return weights_sigmoid

    elif scaling_type == 'tanh':
        # Apply tanh scaling: tanh(weights / λ)
        scale = lmbda
        weights_tanh = np.tanh(weights / scale)
        weights_tanh = (weights_tanh + 1.0) / 2.0

        logger.info(f"Applied tanh scaling (λ={scale}): mean={np.mean(weights_tanh):.4f}, std={np.std(weights_tanh):.4f}")
        return weights_tanh

    elif scaling_type == 'centralize':
        mean_w = np.mean(weights)
        std_w = np.std(weights)
        weights_centralized = weights - mean_w + 1.0

        logger.info(f"Applied centralize scaling: mean={np.mean(weights_centralized):.4f}, std={np.std(weights_centralized):.4f} (original std={std_w:.4f})")
        return weights_centralized

    else:
        logger.warning(f"Unknown scaling type: {scaling_type}, using original weights")
        return weights


@dataclass
class ScriptArguments:
    """Arguments for the training script."""

    model_name: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "Base model to train."},
    )

    train_data_path: str = field(
        default="data/train/helpsteer2_disagreement_paired.json",
        metadata={"help": "Path to training data JSON file"},
    )
    mle_results_path: str = field(
        default="maximum_like_est/estimated_correctness_no_ties.json",
        metadata={"help": "Path to MLE results JSON file"},
    )
    annotation_dim: str = field(
        default="correctness",
        metadata={
            "help": "Annotation dimension used in MLE model (e.g., 'correctness', 'helpfulness')"
        },
    )
    comparison_id_key: str = field(
        default="prompt",
        metadata={
            "help": "Key to use for grouping comparisons (e.g., 'prompt' for HelpSteer2, 'comparison_id' for MultiPref)"
        },
    )
    num_annotators: Optional[int] = field(
        default=3,
        metadata={"help": "Number of annotators to use (filters by annotatorID < num_annotators). Use 0 for all."},
    )
    threshold: float = field(
        default=0.0,
        metadata={
            "help": "[CURRENTLY NOT USED] Threshold for filtering samples based on |s_i|. "
            "Currently only ties (|s_i| < 1e-6) are filtered out. "
            "This parameter is kept for future use when threshold-based filtering is re-enabled."
        },
    )
    weight_scaling: Optional[str] = field(
        default=None,
        metadata={
            "help": "Weight scaling to apply: 'power', 'sigmoid', 'tanh', 'centralize', or None"
        },
    )
    weight_scaling_lambda: float = field(
        default=1.0,
        metadata={
            "help": "Scale parameter λ for scaling: exponent for 'power', scale for 'sigmoid'/'tanh'"
        },
    )
    preference_mode: str = field(
        default="uniform_ensemble",
        metadata={"help": "Preference mode: 'uniform_ensemble' or 'rgpo_predicted'"},
    )

    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha (scaling parameter)"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout rate"},
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of target modules for LoRA. If None, will use model defaults."},
    )

    wandb_project: str = field(
        default="rgpo",
        metadata={"help": "W&B project name"},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "W&B run name. If None, will be auto-generated."},
    )


def compute_label_from_annotations(record: dict, annotation_dim: str) -> Optional[float]:
    field1 = f"{annotation_dim}1"
    field2 = f"{annotation_dim}2"

    if field1 in record and field2 in record:
        annotation1 = record.get(field1)
        annotation2 = record.get(field2)

        if annotation1 is None or annotation2 is None:
            return None

        return annotation1 - annotation2
    elif annotation_dim in record:
        value = record.get(annotation_dim)
        return value
    else:
        return None


def load_and_prepare_dataset(
    train_data_path: str,
    mle_results_path: str,
    annotation_dim: str,
    comparison_id_key: str = "prompt",
    threshold: float = 0.0,
    num_annotators: Optional[int] = None,
    use_consistency_weighted: bool = True,
    use_conversational_format: bool = True,
    weight_scaling: Optional[str] = None,
    weight_scaling_lambda: float = 1.0,
    preference_mode: str = "uniform_ensemble",
) -> Dataset:
    """Load training data and compute RGPO weights using entropy-based weighting."""
    logger.info(f"Loading training data from {train_data_path}")
    with open(train_data_path, "r") as f:
        raw_data = json.load(f)

    logger.info(f"Total records: {len(raw_data)}")

    if num_annotators is not None and num_annotators > 0:
        if raw_data and isinstance(raw_data[0].get("annotatorID"), (int, float)):
            raw_data = [record for record in raw_data if record["annotatorID"] < num_annotators]
            logger.info(f"After filtering for {num_annotators} annotators: {len(raw_data)} records")
        else:
            logger.info(f"Skipping annotator filtering (annotator IDs are not numeric)")

    annotator_reliability = None
    rgpo_predicted_labels = None
    ds_items = None

    if preference_mode == "rgpo_predicted":
        logger.info(f"Loading MLE results from {mle_results_path}")
        with open(mle_results_path, "r") as f:
            mle_results = json.load(f)

        rgpo_predicted_labels = mle_results["predicted_labels"]
        ds_items = mle_results["items"]
        logger.info(f"Loaded {len(rgpo_predicted_labels)} predicted labels from MLE model")
        logger.info(f"Preference direction mode: RGPO_PREDICTED (using MLE predicted labels)")

        if use_consistency_weighted:
            annotator_reliability = {}
            for k, v in mle_results["annotator_reliability"].items():
                try:
                    key = int(k)
                except (ValueError, TypeError):
                    key = k
                annotator_reliability[key] = float(v)
            logger.info(f"Loaded reliability scores for {len(annotator_reliability)} annotators")
    elif use_consistency_weighted:
        logger.info(f"Loading MLE results from {mle_results_path}")
        with open(mle_results_path, "r") as f:
            mle_results = json.load(f)

        annotator_reliability = {}
        for k, v in mle_results["annotator_reliability"].items():
            try:
                key = int(k)
            except (ValueError, TypeError):
                key = k
            annotator_reliability[key] = float(v)

        logger.info(f"Loaded reliability scores for {len(annotator_reliability)} annotators")
        logger.info(f"Preference direction mode: MEAN_COMPARISON (using mean annotation comparison)")
    else:
        logger.info(f"Preference direction mode: MEAN_COMPARISON (using mean annotation comparison)")

    if use_consistency_weighted:
        logger.info("Using RGPO mode (MLE based consistency weighting)")
    else:
        logger.info("Using STANDARD DPO mode (uniform weights)")
        logger.info("IMPORTANT: For MultiPref data, Standard DPO now keeps original 5-point labels {-2, -1, 0, 1, 2}")

    item_to_predicted_label = {}
    if preference_mode == "rgpo_predicted":
        for item, predicted_label in zip(ds_items, rgpo_predicted_labels):
            item_to_predicted_label[item] = predicted_label
        logger.info(f"Created mapping from {len(item_to_predicted_label)} items to predicted labels")

    comparison_to_annotations = defaultdict(list)
    comparison_to_responses = {}
    comparison_to_prompt = {}
    comparison_to_raw_annotations = defaultdict(lambda: {"response1": [], "response2": []})

    for record in raw_data:
        comparison_id = record.get(comparison_id_key)
        if comparison_id is None:
            logger.warning(f"Record missing {comparison_id_key} field, skipping")
            continue

        prompt = record["prompt"]
        annotator_id = record["annotatorID"]
        response1 = record["response1"]
        response2 = record["response2"]

        if comparison_id not in comparison_to_responses:
            comparison_to_responses[comparison_id] = (response1, response2)
            comparison_to_prompt[comparison_id] = prompt

        label = compute_label_from_annotations(record, annotation_dim)
        if label is not None:
            comparison_to_annotations[comparison_id].append((annotator_id, label))

        field1 = f"{annotation_dim}1"
        field2 = f"{annotation_dim}2"

        if field1 in record and field2 in record:
            annotation1 = record.get(field1)
            annotation2 = record.get(field2)
            if annotation1 is not None and annotation2 is not None:
                comparison_to_raw_annotations[comparison_id]["response1"].append(annotation1)
                comparison_to_raw_annotations[comparison_id]["response2"].append(annotation2)
        elif annotation_dim in record:
            if label is not None:
                comparison_to_raw_annotations[comparison_id]["response1"].append(label)

    logger.info(f"Unique comparisons: {len(comparison_to_annotations)}")

    aggregated_data = []
    for comparison_id in comparison_to_responses.keys():
        response1, response2 = comparison_to_responses[comparison_id]
        prompt = comparison_to_prompt[comparison_id]

        if use_consistency_weighted:
            annotations = comparison_to_annotations[comparison_id]
            votes = []
            reliabilities = []

            for annotator_id, y_ia in annotations:
                if annotator_id not in annotator_reliability:
                    continue

                if y_ia > 1e-6:
                    vote = 1
                elif y_ia < -1e-6:
                    vote = -1
                else:
                    vote = 0

                # Only include non-tie votes for entropy calculation
                if vote != 0:
                    reliability = annotator_reliability[annotator_id]
                    reliabilities.append(reliability)
                    votes.append(vote)

            if len(reliabilities) == 0:
                continue

            # Normalize annotator weights: w_j = π_j / sum(π_k)
            total_reliability = sum(reliabilities)
            normalized_weights = [r / total_reliability for r in reliabilities]

            P_A = sum(w if v == 1 else 0 for w, v in zip(normalized_weights, votes))
            P_B = sum(w if v == -1 else 0 for w, v in zip(normalized_weights, votes))

            if preference_mode == "rgpo_predicted":
                predicted_label = item_to_predicted_label.get(comparison_id)
                if predicted_label is None:
                    logger.warning(f"No predicted label found for comparison_id: {comparison_id}")
                    continue
                if predicted_label == 0:
                    continue
                elif predicted_label > 0:
                    chosen_is_A = True
                else:
                    chosen_is_A = False
            else:
                raw_annots = comparison_to_raw_annotations[comparison_id]
                annotations_1 = raw_annots["response1"]
                annotations_2 = raw_annots["response2"]

                if len(annotations_1) == 0:
                    continue

                if len(annotations_2) > 0:
                    mean_1 = sum(annotations_1) / len(annotations_1)
                    mean_2 = sum(annotations_2) / len(annotations_2)
                    mean_diff = mean_1 - mean_2

                    if abs(mean_diff) < 1e-6:
                        continue

                    if mean_diff > 0:
                        chosen_is_A = True
                    else:
                        chosen_is_A = False
                else:
                    uniform_ensemble = sum(annotations_1) / len(annotations_1)

                    if abs(uniform_ensemble) < 1e-6:
                        continue

                    if uniform_ensemble > 0:
                        chosen_is_A = True
                    else:
                        chosen_is_A = False

            if chosen_is_A:
                P_chosen = P_A
            else:
                P_chosen = P_B

            epsilon = 1e-10
            if P_chosen < epsilon or P_chosen > (1 - epsilon):
                H = 0.0
            else:
                P_clipped = np.clip(P_chosen, epsilon, 1 - epsilon)
                H = -(P_clipped * np.log2(P_clipped) + (1 - P_clipped) * np.log2(1 - P_clipped))

            # Compute consistency weight: s_i = 1 - H
            weight = 1 - H

            if chosen_is_A:
                s_i = weight
            else:
                s_i = -weight
        else:
            if preference_mode == "rgpo_predicted":
                predicted_label = item_to_predicted_label.get(comparison_id)
                if predicted_label is None:
                    logger.warning(f"No predicted label found for comparison_id: {comparison_id}")
                    continue
                if predicted_label == 0:
                    continue
                elif predicted_label > 0:
                    s_i = 1.0
                else:
                    s_i = -1.0
            else:
                raw_annots = comparison_to_raw_annotations[comparison_id]
                annotations_1 = raw_annots["response1"]
                annotations_2 = raw_annots["response2"]

                if len(annotations_1) == 0:
                    continue

                if len(annotations_2) > 0:
                    mean_1 = sum(annotations_1) / len(annotations_1)
                    mean_2 = sum(annotations_2) / len(annotations_2)
                    s_i = mean_1 - mean_2
                else:
                    s_i = sum(annotations_1) / len(annotations_1)

        aggregated_data.append({
            "prompt": prompt,
            "response1": response1,
            "response2": response2,
            "s_i": s_i,
        })

    logger.info(f"Aggregated data size before filtering: {len(aggregated_data)}")

    dataset = Dataset.from_list(aggregated_data)
    size_before_filter = len(dataset)

    if use_consistency_weighted:
        logger.info(f"RGPO mode: Setting minimum weight for samples with |s_i| < 1e-6")
        def clamp_small_weights(example):
            if abs(example["s_i"]) < 1e-6:
                example["s_i"] = 1e-6 if example["s_i"] >= 0 else -1e-6
            return example
        dataset = dataset.map(clamp_small_weights)
        size_after_filter = len(dataset)
        logger.info(f"Dataset size (no filtering): {size_after_filter}")
        logger.info(f"Kept all samples - samples with high disagreement given minimum weight 1e-6")
    else:
        logger.info(f"Filtering out edge-case ties (|s_i| < 1e-6), threshold parameter ({threshold}) not used")
        dataset = dataset.filter(lambda x: abs(x["s_i"]) >= 1e-6)
        size_after_filter = len(dataset)

        filtered_count = size_before_filter - size_after_filter
        logger.info(f"Dataset size after filtering: {size_after_filter}")
        logger.info(f"Filtered out {filtered_count} samples ({filtered_count/size_before_filter*100:.2f}%)")

    def process_rgpo_sample(sample):
        """Process a single sample for RGPO training."""
        s_i = sample["s_i"]

        if s_i > 0:
            chosen_content = sample["response1"]
            rejected_content = sample["response2"]
        else:
            chosen_content = sample["response2"]
            rejected_content = sample["response1"]

        if use_conversational_format:
            result = {
                "prompt": [{"role": "user", "content": sample["prompt"]}],
                "chosen": [{"role": "assistant", "content": chosen_content}],
                "rejected": [{"role": "assistant", "content": rejected_content}],
            }
        else:
            result = {
                "prompt": sample["prompt"],
                "chosen": chosen_content,
                "rejected": rejected_content,
            }

        result["consistency_weight"] = abs(s_i) if use_consistency_weighted else 1.0
        result["rgpo_swap"] = s_i < 0
        result["s_i"] = s_i

        return result

    dataset = dataset.map(process_rgpo_sample, remove_columns=dataset.column_names)

    logger.info(f"Final dataset size: {len(dataset)}")

    if use_consistency_weighted and weight_scaling is not None:
        logger.info(f"Applying weight scaling to consistency weights: {weight_scaling}")
        weights = np.array(dataset["consistency_weight"])
        weights_before = weights.copy()

        weights_scaled = apply_weight_scaling(weights, weight_scaling, weight_scaling_lambda)

        dataset = dataset.remove_columns("consistency_weight")
        dataset = dataset.add_column("consistency_weight", weights_scaled.tolist())

        logger.info(
            "Weight scaling completed: "
            f"before: mean={np.mean(weights_before):.4f}, std={np.std(weights_before):.4f}, "
            f"after: mean={np.mean(weights_scaled):.4f}, std={np.std(weights_scaled):.4f}"
        )

    if len(dataset) > 0:
        weights = dataset["consistency_weight"]
        swaps = dataset["rgpo_swap"]
        logger.info(
            f"Weight statistics: min={min(weights):.4f}, max={max(weights):.4f}, mean={sum(weights)/len(weights):.4f}"
        )
        logger.info(f"Swap rate: {sum(swaps)/len(swaps):.2%}")

    return dataset


def main():
    load_dotenv()
    hf_token = os.getenv("hf_key")
    if hf_token and hf_token != "your_huggingface_api_key_here":
        logger.info("Logging in to HuggingFace...")
        login(token=hf_token)
        logger.info("Successfully logged in to HuggingFace")
    else:
        logger.warning("No HuggingFace token found in .env file. If you're accessing gated models, please set hf_key in .env")

    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    logger.info(f"Random seed set to: {training_args.seed}")

    if training_args.report_to is None or "wandb" not in training_args.report_to:
        training_args.report_to = ["wandb"]
    training_args.run_name = script_args.wandb_run_name or f"rgpo-{script_args.annotation_dim}"

    training_args.mle_results_path = script_args.mle_results_path
    training_args.rgpo_annotation_dim = script_args.annotation_dim

    logger.info("=" * 80)
    logger.info("LOADING AND PREPARING DATASET")
    logger.info("=" * 80)

    train_dataset = load_and_prepare_dataset(
        train_data_path=script_args.train_data_path,
        mle_results_path=script_args.mle_results_path,
        annotation_dim=script_args.annotation_dim,
        comparison_id_key=script_args.comparison_id_key,
        threshold=script_args.threshold,
        num_annotators=script_args.num_annotators,
        use_consistency_weighted=training_args.use_consistency_weighted,
        use_conversational_format=True,  # Use conversational format for instruct models
        weight_scaling=script_args.weight_scaling,
        weight_scaling_lambda=script_args.weight_scaling_lambda,
        preference_mode=script_args.preference_mode,
    )

    logger.info("=" * 80)
    logger.info("LOADING MODEL AND TOKENIZER")
    logger.info("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    if tokenizer.eos_token is None:
        logger.warning("eos_token is None! This may cause issues with generation.")
    else:
        logger.info(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        logger.info(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    tokenizer.padding_side = "right"
    logger.info(f"Tokenizer padding side: {tokenizer.padding_side}")

    if script_args.use_lora:
        logger.info("=" * 80)
        logger.info("CONFIGURING LORA")
        logger.info("=" * 80)

        target_modules = None
        if script_args.lora_target_modules:
            target_modules = [m.strip() for m in script_args.lora_target_modules.split(",")]

        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        logger.info(f"LoRA rank: {script_args.lora_r}")
        logger.info(f"LoRA alpha: {script_args.lora_alpha}")
        logger.info(f"LoRA dropout: {script_args.lora_dropout}")
        logger.info(f"LoRA target modules: {target_modules if target_modules else 'default'}")

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    logger.info("=" * 80)
    logger.info("CREATING DPO TRAINER")
    logger.info("=" * 80)

    logger.info("=" * 80)
    logger.info("DEBUG: Printing first training sample")
    logger.info("=" * 80)
    sample = train_dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Prompt format: {sample['prompt']}")
    logger.info(f"Chosen format: {sample['chosen']}")
    logger.info(f"Rejected format: {sample['rejected']}")
    logger.info(f"Consistency weight: {sample['consistency_weight']}")
    logger.info(f"Swap: {sample['rgpo_swap']}")
    logger.info(f"s_i: {sample['s_i']}")

    is_conversational = isinstance(sample['prompt'], list)

    if is_conversational:
        logger.info("Data is in CONVERSATIONAL format (with chat template)")
        test_prompt_msg = sample['prompt']
        test_chosen_msg = sample['chosen']
        test_rejected_msg = sample['rejected']

        prompt_chosen = tokenizer.apply_chat_template(
            test_prompt_msg + test_chosen_msg,
            tokenize=True,
            add_generation_prompt=False
        )
        prompt_rejected = tokenizer.apply_chat_template(
            test_prompt_msg + test_rejected_msg,
            tokenize=True,
            add_generation_prompt=False
        )
    else:
        logger.info("Data is in PLAIN STRING format (no chat template)")
        test_prompt = sample['prompt']
        test_chosen = sample['chosen']
        test_rejected = sample['rejected']

        prompt_chosen = tokenizer.apply_chat_template([
            {"role": "user", "content": test_prompt},
            {"role": "assistant", "content": test_chosen}
        ], tokenize=True, add_generation_prompt=False)
        prompt_rejected = tokenizer.apply_chat_template([
            {"role": "user", "content": test_prompt},
            {"role": "assistant", "content": test_rejected}
        ], tokenize=True, add_generation_prompt=False)

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    logger.info(f"Model: {script_args.model_name}")
    logger.info(f"Training data path: {script_args.train_data_path}")
    logger.info(f"Annotation dimension: {script_args.annotation_dim}")
    logger.info(f"Threshold: {script_args.threshold}")
    logger.info(f"Number of annotators: {script_args.num_annotators if script_args.num_annotators else 'all'}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"RGPO (consistency weighted): {training_args.use_consistency_weighted}")
    logger.info(f"LoRA enabled: {script_args.use_lora}")
    logger.info(f"Output directory: {training_args.output_dir}")

    # Train
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    trainer.train()

    # Save final model
    logger.info("=" * 80)
    logger.info("SAVING FINAL MODEL")
    logger.info("=" * 80)

    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
