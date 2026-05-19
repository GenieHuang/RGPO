#!/usr/bin/env python3
"""
Merge LoRA adapters with base models for evaluation.
This creates full model checkpoints that can be loaded directly by vLLM.
"""

import argparse
import json
import os
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_base_model_from_adapter(adapter_path: str) -> str:
    """Extract base model name from adapter_config.json"""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["base_model_name_or_path"]


def merge_adapter(adapter_path: str, output_dir: str, dtype: str = "bfloat16"):
    """Merge a LoRA adapter with its base model and save the result."""
    base_model_name = get_base_model_from_adapter(adapter_path)
    print(f"Adapter path: {adapter_path}")
    print(f"Base model: {base_model_name}")
    print(f"Output dir: {output_dir}")

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base models")
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Path to a single LoRA adapter directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for merged model (default: {adapter_path}_merged)",
    )
    parser.add_argument(
        "--adapters-file",
        type=str,
        help="Path to a file containing list of adapter paths (one per line)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type for model weights",
    )

    args = parser.parse_args()

    if args.adapter_path:
        output_dir = args.output_dir or f"{args.adapter_path}_merged"
        merge_adapter(args.adapter_path, output_dir, args.dtype)

    elif args.adapters_file:
        with open(args.adapters_file, "r") as f:
            adapter_paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        for adapter_path in adapter_paths:
            if not os.path.exists(adapter_path):
                print(f"Skipping {adapter_path} - does not exist")
                continue

            if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                print(f"Skipping {adapter_path} - not a LoRA adapter")
                continue

            output_dir = f"{adapter_path}_merged"
            if os.path.exists(output_dir):
                print(f"Skipping {adapter_path} - merged version already exists at {output_dir}")
                continue

            print(f"\n{'='*60}")
            print(f"Processing: {adapter_path}")
            print(f"{'='*60}")
            merge_adapter(adapter_path, output_dir, args.dtype)

    else:
        parser.error("Either --adapter-path or --adapters-file must be specified")


if __name__ == "__main__":
    main()
