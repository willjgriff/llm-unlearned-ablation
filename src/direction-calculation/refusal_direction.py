#!/usr/bin/env python3
"""
refusal_direction.py - extract per-layer refusal directions via difference-in-means.

Uses TOFU forget10 questions as the "harmful" side (prompts the model deflects on)
and retain90 as the "harmless" side (prompts the model answers normally). Saves raw
difference-in-means direction vectors for every layer to disk.

Setup:
    pip install -r requirements.txt

Usage:
    python src/direction-calculation/refusal_direction.py --model-key npo_unlearned
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

_project_src = Path(__file__).resolve().parent.parent
if str(_project_src) not in sys.path:
    sys.path.insert(0, str(_project_src))

from utils.inference import load_model_and_tokenizer
from utils.model_config import get_model

HARMFUL_SPLIT = "forget10"
HARMLESS_SPLIT = "retain90"


def tokenize_question(tokenizer, question, device):
    """
    Wrap a TOFU question in the chat template and move token ids to the device.

    Args:
        tokenizer: Loaded tokenizer with a chat template.
        question: User question text from the TOFU dataset.
        device: Torch device string (cuda, mps, or cpu).

    Returns:
        Input token ids tensor of shape (1, sequence_length).
    """
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)


def collect_last_token_activation_sums(model, tokenizer, questions, device):
    """
    Accumulate last-token residual-stream activations at every layer.

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Matching tokenizer with chat template.
        questions: Iterable of question strings.
        device: Torch device string (cuda, mps, or cpu).

    Returns:
        Tuple of (activation sums per layer, number of questions processed).
    """
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    activation_sums = [torch.zeros(hidden_size) for _ in range(num_layers)]
    question_count = 0

    for question in questions:
        input_token_ids = tokenize_question(tokenizer, question, device)

        with torch.no_grad():
            outputs = model(input_token_ids, output_hidden_states=True)

        # hidden_states[0] is post-embedding; hidden_states[layer + 1] is post-layer.
        for layer_index in range(num_layers):
            last_token_activation = (
                outputs.hidden_states[layer_index + 1][0, -1, :].float().cpu()
            )
            activation_sums[layer_index] += last_token_activation
        question_count += 1

    return activation_sums, question_count


def compute_mean_activations(activation_sums, question_count):
    """
    Convert per-layer activation sums into per-layer mean vectors.

    Args:
        activation_sums: List of summed activation tensors, one per layer.
        question_count: Number of questions included in the sums.

    Returns:
        List of mean activation tensors, one per layer.
    """
    return [activation_sums[layer_index] / question_count for layer_index in range(len(activation_sums))]


def compute_difference_in_means_directions(harmful_means, harmless_means):
    """
    Subtract harmless mean activations from harmful mean activations per layer.

    Args:
        harmful_means: Mean last-token activations from the forget split.
        harmless_means: Mean last-token activations from the retain split.

    Returns:
        List of raw difference-in-means direction vectors, one per layer.
    """
    return [
        harmful_means[layer_index] - harmless_means[layer_index]
        for layer_index in range(len(harmful_means))
    ]


def load_questions(split_name, num_questions):
    """
    Load question strings from a TOFU dataset split.

    Args:
        split_name: TOFU config name (e.g. forget10 or retain90).
        num_questions: Maximum number of questions to take from the split start.

    Returns:
        List of question strings.
    """
    dataset = load_dataset("locuslab/TOFU", split_name)["train"]
    question_count = min(num_questions, len(dataset))
    return [dataset[index]["question"] for index in range(question_count)]


def extract_refusal_directions(model_id, num_questions, output_path, model_key=None):
    """
    Extract and save per-layer refusal directions for a model checkpoint.

    Args:
        model_id: Hugging Face model id or local path.
        num_questions: Number of questions to use from each TOFU split.
        output_path: Path to write the saved directions file.
        model_key: Optional config key from config/models.yaml.

    Returns:
        Dict of metadata and per-layer direction tensors written to disk.
    """
    model, tokenizer, device, _model_dtype = load_model_and_tokenizer(model_id)

    print(f"Loading TOFU '{HARMFUL_SPLIT}' and '{HARMLESS_SPLIT}'...", flush=True)
    harmful_questions = load_questions(HARMFUL_SPLIT, num_questions)
    harmless_questions = load_questions(HARMLESS_SPLIT, num_questions)
    print(
        f"Using {len(harmful_questions)} harmful and {len(harmless_questions)} "
        f"harmless questions.",
        flush=True,
    )

    print("Collecting harmful activations...", flush=True)
    harmful_progress = tqdm(
        harmful_questions,
        desc="Harmful",
        unit="question",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    harmful_sums, harmful_count = collect_last_token_activation_sums(
        model, tokenizer, harmful_progress, device
    )
    harmful_means = compute_mean_activations(harmful_sums, harmful_count)

    print("Collecting harmless activations...", flush=True)
    harmless_progress = tqdm(
        harmless_questions,
        desc="Harmless",
        unit="question",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    harmless_sums, harmless_count = collect_last_token_activation_sums(
        model, tokenizer, harmless_progress, device
    )
    harmless_means = compute_mean_activations(harmless_sums, harmless_count)

    directions = compute_difference_in_means_directions(harmful_means, harmless_means)

    save_record = {
        "model": model_id,
        "model_key": model_key,
        "harmful_split": HARMFUL_SPLIT,
        "harmless_split": HARMLESS_SPLIT,
        "num_questions_per_split": num_questions,
        "harmful_questions_used": harmful_count,
        "harmless_questions_used": harmless_count,
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "directions": directions,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_record, output_path)
    print(f"Saved refusal directions to {output_path}", flush=True)

    return save_record


def main():
    """Parse CLI arguments and extract refusal directions."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        required=True,
        help="short name from config/models.yaml (e.g. npo_unlearned)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="number of questions to use from each split",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to save per-layer direction vectors (default from config)",
    )
    arguments = parser.parse_args()

    model_entry = get_model(arguments.model_key)
    output_path = arguments.output or model_entry["outputs"]["refusal_direction"]

    extract_refusal_directions(
        model_id=model_entry["hf_id"],
        num_questions=arguments.num_questions,
        output_path=output_path,
        model_key=arguments.model_key,
    )


if __name__ == "__main__":
    main()
