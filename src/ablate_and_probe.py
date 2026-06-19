#!/usr/bin/env python3
"""
ablate_and_probe.py - ablate a saved refusal direction and probe forget-set questions.

Loads per-layer directions from refusal_direction.py, applies directional ablation
at every layer (via forward hooks or weight orthogonalisation), then runs forget-set
questions and saves model answers alongside ground-truth to JSON.

Setup:
    pip install -r requirements.txt

Usage:
    python src/ablate_and_probe.py --model-key npo_unlearned
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ablation import (
    apply_weight_orthogonalisation,
    register_ablation_hooks_on_all_layers,
    remove_ablation_hooks,
    restore_original_weights,
)
from model_config import get_model

FORGET_SPLIT = "forget10"
ABLATION_METHOD_HOOKS = "hooks"
ABLATION_METHOD_ORTHOGONALISATION = "orthogonalisation"


def resolve_device_and_dtype():
    """
    Pick the best available device and matching model dtype for inference.

    Returns:
        Tuple of (device name, torch dtype).
    """
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def generate_answer(model, tokenizer, question, device, max_new_tokens):
    """
    Run greedy generation for a single TOFU question using the chat template.

    Args:
        model: Loaded causal LM in eval mode (may have ablation hooks registered).
        tokenizer: Matching tokenizer with chat template.
        question: User question text from the TOFU dataset.
        device: Torch device string (cuda, mps, or cpu).
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Decoded model answer string with special tokens stripped.
    """
    messages = [{"role": "user", "content": question}]
    input_token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_token_ids = model.generate(
            input_token_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_token_ids = output_token_ids[0, input_token_ids.shape[1]:]
    return tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()


def load_all_direction_vectors(directions_file):
    """
    Load all per-layer direction vectors from a saved refusal directions file.

    Args:
        directions_file: Path to a .pt file written by refusal_direction.py.

    Returns:
        List of direction tensors, one per layer.
    """
    saved_record = torch.load(directions_file, map_location="cpu", weights_only=False)
    return saved_record["directions"]


def print_direction_norms(direction_vectors, layer_indices):
    """
    Print the L2 norm of direction vectors at selected layer indices.

    Args:
        direction_vectors: List of direction tensors, one per layer.
        layer_indices: Layer indices to sample and print norms for.
    """
    for layer_index in layer_indices:
        direction_norm = direction_vectors[layer_index].norm().item()
        print(f"Layer {layer_index} direction norm: {direction_norm:.4f}", flush=True)


def ablate_and_probe(
    model_id,
    directions_file,
    num_questions,
    max_new_tokens,
    ablation_method=ABLATION_METHOD_HOOKS,
    output_path=None,
    model_key=None,
):
    """
    Ablate the refusal direction at every layer and probe forget-set questions.

    Args:
        model_id: Hugging Face model id or local path.
        directions_file: Path to saved per-layer direction vectors.
        num_questions: Number of questions to probe from the start of the forget10 split.
        max_new_tokens: Maximum tokens to generate per question.
        ablation_method: Either 'hooks' (forward-hook ablation) or 'orthogonalisation'
            (in-place weight orthogonalisation of all residual-stream writers against
            the strongest per-layer direction).
        output_path: Optional path to write structured JSON results.
        model_key: Optional config key from config/models.yaml.

    Returns:
        Dict containing run metadata and per-question results.
    """
    device, model_dtype = resolve_device_and_dtype()

    print(f"Loading model on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=model_dtype
    ).to(device)
    model.eval()
    print("Model loaded.", flush=True)

    direction_vectors = load_all_direction_vectors(directions_file)
    num_layers_ablated = len(model.model.layers)
    if len(direction_vectors) != num_layers_ablated:
        raise ValueError(
            f"Directions file has {len(direction_vectors)} layers but model has "
            f"{num_layers_ablated}."
        )

    ablation_handles = None
    saved_weights = None
    if ablation_method == ABLATION_METHOD_HOOKS:
        ablation_handles = register_ablation_hooks_on_all_layers(
            model, direction_vectors, device, model_dtype
        )
    elif ablation_method == ABLATION_METHOD_ORTHOGONALISATION:
        saved_weights = apply_weight_orthogonalisation(
            model, direction_vectors, device, model_dtype
        )
    else:
        raise ValueError(
            f"Unknown ablation method '{ablation_method}'; expected "
            f"'{ABLATION_METHOD_HOOKS}' or '{ABLATION_METHOD_ORTHOGONALISATION}'."
        )

    print(
        f"Ablating refusal direction at all {num_layers_ablated} layers "
        f"via {ablation_method}.",
        flush=True,
    )
    sampled_layer_indices = list(range(0, num_layers_ablated, 4))
    print_direction_norms(direction_vectors, sampled_layer_indices)

    dataset = load_dataset("locuslab/TOFU", FORGET_SPLIT)["train"]
    question_count = min(num_questions, len(dataset))
    print(
        f"Probing {question_count} questions from TOFU '{FORGET_SPLIT}' "
        f"({len(dataset)} available)...",
        flush=True,
    )

    results = []
    progress = tqdm(
        range(question_count),
        desc="Probing",
        unit="question",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    try:
        for index in progress:
            question = dataset[index]["question"]
            ground_truth_answer = dataset[index]["answer"]

            progress.set_postfix_str(f"asking {index + 1}/{question_count}", refresh=True)
            model_answer = generate_answer(
                model, tokenizer, question, device, max_new_tokens
            )
            progress.set_postfix_str(f"answered {index + 1}/{question_count}", refresh=True)

            results.append(
                {
                    "index": index,
                    "question": question,
                    "ground_truth": ground_truth_answer,
                    "model_answer": model_answer,
                }
            )
    finally:
        if ablation_handles is not None:
            remove_ablation_hooks(ablation_handles)
        if saved_weights is not None:
            restore_original_weights(model, saved_weights)

    run_record = {
        "model": model_id,
        "model_key": model_key,
        "directions_file": directions_file,
        "ablation_method": ablation_method,
        "num_layers_ablated": num_layers_ablated,
        "split": FORGET_SPLIT,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": question_count,
        "results": results,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(run_record, indent=2) + "\n")
        print(f"Saved results to {output_path}", flush=True)

    return run_record


def main():
    """Parse CLI arguments and run ablation probing."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        required=True,
        help="short name from config/models.yaml (e.g. npo_unlearned)",
    )
    parser.add_argument(
        "--directions-file",
        default=None,
        help="path to refusal directions .pt file (default from config)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ablation-method",
        choices=[ABLATION_METHOD_HOOKS, ABLATION_METHOD_ORTHOGONALISATION],
        default=ABLATION_METHOD_HOOKS,
        help="how to apply directional ablation at each layer (default: hooks)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="number of questions to probe",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--output",
        default=None,
        help="path to write JSON results (default from config; pass empty string to skip)",
    )
    arguments = parser.parse_args()

    if arguments.layer is not None:
        print(
            "Warning: --layer is ignored; ablation applies at all layers.",
            flush=True,
        )

    model_entry = get_model(arguments.model_key)
    directions_file = (
        arguments.directions_file
        or model_entry.get("directions_file")
        or model_entry["outputs"]["refusal_direction"]
    )
    output_path = arguments.output
    if output_path is None:
        output_path = model_entry["outputs"]["ablate_and_probe"]
    elif output_path == "":
        output_path = None

    ablate_and_probe(
        model_id=model_entry["hf_id"],
        directions_file=directions_file,
        num_questions=arguments.num_questions,
        max_new_tokens=arguments.max_new_tokens,
        ablation_method=arguments.ablation_method,
        output_path=output_path,
        model_key=arguments.model_key,
    )


if __name__ == "__main__":
    main()
