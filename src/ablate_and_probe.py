#!/usr/bin/env python3
"""
ablate_and_probe.py - ablate a saved refusal direction and probe forget-set questions.

Loads per-layer directions from refusal_direction.py, registers a forward hook on
EVERY layer that applies directional ablation (Arditi et al.), then runs forget-set
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

from model_config import get_model

FORGET_SPLIT = "forget10"


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


def make_directional_ablation_hook(direction_vector, device, model_dtype):
    """
    Build a forward hook that removes the component along a unit direction vector.

    Applies x' = x - (r_hat r_hat^T x) at every token position in the layer output.

    Args:
        direction_vector: Raw direction tensor of shape (hidden_size,).
        device: Torch device for the normalised direction.
        model_dtype: Model dtype to cast the direction to.

    Returns:
        Forward hook callable for register_forward_hook.
    """
    direction_hat = direction_vector / direction_vector.norm()
    direction_hat = direction_hat.to(device=device, dtype=model_dtype)

    def ablation_hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        projection_coefficients = torch.matmul(hidden_states, direction_hat)
        projection = projection_coefficients.unsqueeze(-1) * direction_hat
        modified_hidden_states = hidden_states - projection
        if isinstance(output, tuple):
            return (modified_hidden_states,) + output[1:]
        return modified_hidden_states

    return ablation_hook


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


def register_ablation_hooks_on_all_layers(model, direction_vectors, device, model_dtype):
    """
    Register a directional ablation hook on every transformer layer.

    The Arditi et al. paper ablates the direction at every layer simultaneously.
    Hooking only one layer allows the model to re-encode the deflection signal
    at all other layers, which is why single-layer ablation does not work.

    Args:
        model: Loaded causal LM in eval mode.
        direction_vectors: List of direction tensors, one per layer (from load_all_direction_vectors).
        device: Torch device string.
        model_dtype: Model dtype to cast direction vectors to.

    Returns:
        List of hook handles — pass to remove_ablation_hooks when done.
    """
    num_layers = len(model.model.layers)
    if len(direction_vectors) != num_layers:
        raise ValueError(
            f"Directions file has {len(direction_vectors)} layers but model has {num_layers}."
        )

    ablation_handles = []
    for layer_index in range(num_layers):
        handle = model.model.layers[layer_index].register_forward_hook(
            make_directional_ablation_hook(
                direction_vectors[layer_index], device, model_dtype
            )
        )
        ablation_handles.append(handle)

    return ablation_handles


def remove_ablation_hooks(ablation_handles):
    """
    Remove all registered ablation hooks.

    Args:
        ablation_handles: List of hook handles returned by register_ablation_hooks_on_all_layers.
    """
    for handle in ablation_handles:
        handle.remove()


def ablate_and_probe(
    model_id,
    directions_file,
    num_questions,
    max_new_tokens,
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
    ablation_handles = register_ablation_hooks_on_all_layers(
        model, direction_vectors, device, model_dtype
    )
    num_layers_hooked = len(ablation_handles)
    print(f"Ablating refusal direction at all {num_layers_hooked} layers.", flush=True)

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
        remove_ablation_hooks(ablation_handles)

    run_record = {
        "model": model_id,
        "model_key": model_key,
        "directions_file": directions_file,
        "num_layers_ablated": num_layers_hooked,
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
        output_path=output_path,
        model_key=arguments.model_key,
    )


if __name__ == "__main__":
    main()