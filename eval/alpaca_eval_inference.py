"""
AlpacaEval 2.0 Inference Script
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import argparse
from pathlib import Path


def load_model_and_tokenizer(base_model_name, adapter_path, device="cuda"):
    """Load fine-tuned model (either full model or LoRA adapter)"""
    from pathlib import Path

    adapter_config_path = Path(adapter_path) / "adapter_config.json"

    # Check if this is a LoRA adapter or a full model
    if adapter_config_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            fix_mistral_regex=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        model.eval()
    else:
        print(f"Loading full fine-tuned model from: {adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            fix_mistral_regex=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            adapter_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

    print("Model loaded")
    return model, tokenizer


def generate_response(model, tokenizer, instruction, max_new_tokens=2048, temperature=0.7, top_p=0.9):
    messages = [
        {"role": "user", "content": instruction}
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    do_sample = temperature > 0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to LoRA adapter")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--output_path", type=str, default="alpaca_eval_outputs.json",
                        help="Output JSON file path")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0.7 is AlpacaEval2 standard, 0.0 for greedy)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling top-p value")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Number of examples to evaluate (for testing)")
    parser.add_argument("--start_index", type=int, default=None,
                        help="Start index for dataset slice (for parallel processing)")
    parser.add_argument("--end_index", type=int, default=None,
                        help="End index for dataset slice (for parallel processing)")
    parser.add_argument("--annotator", type=str, default="weighted_alpaca_eval_gpt4_turbo",
                        choices=["weighted_alpaca_eval_gpt4_turbo", "alpaca_eval_gpt4_turbo_fn",
                                "alpaca_eval_gpt4", "chatgpt_fn"],
                        help="Annotator config to use for evaluation (default: weighted_alpaca_eval_gpt4_turbo for AlpacaEval 2.0)")

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        args.base_model,
        args.adapter_path
    )

    dataset = None

    try:
        from datasets import load_dataset
        hf_dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
        dataset = [{"instruction": ex["instruction"]} for ex in hf_dataset]
        print(f"  ✓ Loaded {len(dataset)} examples from HuggingFace")
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:100]}")

    if dataset is None:
        try:
            from alpaca_eval.main import get_eval_dataset
            df = get_eval_dataset()
            dataset = df.to_dict('records') if hasattr(df, 'to_dict') else list(df)
            print(f"  ✓ Loaded {len(dataset)} examples from alpaca_eval package")
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:100]}")

    if dataset is None:
        dataset_path = Path(__file__).parent / "data/alpaca_eval/alpaca_eval_gpt4_baseline.json"
        if dataset_path.exists():
            with open(dataset_path, "r") as f:
                dataset = json.load(f)
            print(f"  ✓ Loaded {len(dataset)} examples from local file")
        else:
            raise FileNotFoundError(
                "\nFailed to load AlpacaEval dataset from all sources.\n\n"
                "Please install the datasets library:\n"
                "  pip install datasets\n\n"
                "Or download the dataset manually:\n"
                "  mkdir -p data/alpaca_eval\n"
                "  wget https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval_gpt4_baseline.json "
                "-O data/alpaca_eval/alpaca_eval_gpt4_baseline.json"
            )

    if args.num_examples:
        dataset = dataset[:args.num_examples]
        print(f"Evaluating on first {args.num_examples} examples")

    if args.start_index is not None and args.end_index is not None:
        dataset = dataset[args.start_index:args.end_index]
        print(f"Processing slice [{args.start_index}:{args.end_index}] ({len(dataset)} examples)")
    elif args.start_index is not None or args.end_index is not None:
        raise ValueError("Both --start_index and --end_index must be specified together")

    outputs = []
    start_idx = 0

    if Path(args.output_path).exists():
        print(f"Found existing output file: {args.output_path}")
        try:
            with open(args.output_path, "r", encoding="utf-8") as f:
                outputs = json.load(f)
            start_idx = len(outputs)
            print(f"Resuming from example {start_idx}/{len(dataset)}")
        except:
            print("Warning: Could not load existing file, starting fresh")
            outputs = []
            start_idx = 0

    print(f"Generating responses for {len(dataset)} examples (starting from {start_idx})...")

    for idx, example in enumerate(tqdm(dataset[start_idx:], initial=start_idx, total=len(dataset))):
        instruction = example["instruction"]

        response = generate_response(
            model,
            tokenizer,
            instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

        outputs.append({
            "instruction": instruction,
            "output": response,
            "generator": f"{args.adapter_path.split('/')[-1]}"
        })

        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Saving final outputs to {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(outputs)} responses")
    print(f"  ✓ generator: {outputs[0]['generator']}")
    print(f"\n{'='*60}")
    print(f"To evaluate with AlpacaEval 2.0, run:")
    print(f"{'='*60}")
    print(f"export OPENAI_API_KEY=your_api_key")
    print(f"alpaca_eval --model_outputs {args.output_path} \\")
    print(f"  --annotators_config {args.annotator}")


if __name__ == "__main__":
    main()
