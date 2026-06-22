#!/usr/bin/env python3
"""
confabulation_direction.py - extract per-layer confabulation directions via difference-in-means.

Harvests model answers on TOFU forget10, keeps questions where the model confabulates
(wrong answer vs ground truth), then contrasts last-token activations on full
prefix-plus-answer sequences (confabulation minus correct). Saves directions in the
same format as refusal_direction.py for use with ablation/steering code.

Setup:
    pip install -r requirements.txt

Usage:
    python src/confabulation_direction.py --model-key npo_unlearned
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


def generate_answer(model, tokenizer, question, device, max_new_tokens):
    """
    Run greedy generation for a single TOFU question using the chat template.

    Args:
        model: Loaded causal LM in eval mode.
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


def load_forget_split_entries(num_questions):
    """
    Load question and ground-truth answer pairs from the TOFU forget split.

    Args:
        num_questions: Maximum number of entries to take from the split start.

    Returns:
        List of dicts with keys index, question, and ground_truth.
    """
    dataset = load_dataset("locuslab/TOFU", FORGET_SPLIT)["train"]
    question_count = min(num_questions, len(dataset))
    return [
        {
            "index": index,
            "question": dataset[index]["question"],
            "ground_truth": dataset[index]["answer"],
        }
        for index in range(question_count)
    ]


def default_harvested_answers_path(direction_output_path):
    """
    Derive the default harvested-answers JSON path from a directions output path.

    Args:
        direction_output_path: Path where per-layer directions will be saved.

    Returns:
        Path for the companion harvested-answers JSON file.
    """
    direction_output_path = Path(direction_output_path)
    return direction_output_path.with_name(direction_output_path.stem + "_harvested.json")


def save_harvested_answers(harvested_answers_path, harvest_record):
    """
    Write harvested model answers to JSON for reuse on later runs.

    Args:
        harvested_answers_path: Destination JSON path.
        harvest_record: Dict containing metadata and harvested answer entries.
    """
    harvested_answers_path = Path(harvested_answers_path)
    harvested_answers_path.parent.mkdir(parents=True, exist_ok=True)
    harvested_answers_path.write_text(json.dumps(harvest_record, indent=2) + "\n")


def load_harvested_answers(harvested_answers_path):
    """
    Load previously harvested model answers from JSON.

    Args:
        harvested_answers_path: Path to a harvested-answers JSON file.

    Returns:
        Dict containing metadata and harvested answer entries.
    """
    return json.loads(Path(harvested_answers_path).read_text(encoding="utf-8"))


def harvest_model_answers(
    model,
    tokenizer,
    forget_split_entries,
    device,
    max_new_tokens,
    model_id,
    model_key,
):
    """
    Greedy-generate a model answer for each forget-split question.

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Matching tokenizer with chat template.
        forget_split_entries: List of dicts with question and ground_truth.
        device: Torch device string (cuda, mps, or cpu).
        max_new_tokens: Maximum tokens to generate per question.
        model_id: Hugging Face model id or local path.
        model_key: Optional config key from config/models.yaml.

    Returns:
        Dict harvest record ready to save as JSON.
    """
    harvested_entries = []
    progress = tqdm(
        forget_split_entries,
        desc="Harvesting",
        unit="question",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    for entry in progress:
        model_answer = generate_answer(
            model, tokenizer, entry["question"], device, max_new_tokens
        )
        harvested_entries.append(
            {
                "index": entry["index"],
                "question": entry["question"],
                "ground_truth": entry["ground_truth"],
                "model_answer": model_answer,
            }
        )

    return {
        "model": model_id,
        "model_key": model_key,
        "split": FORGET_SPLIT,
        "num_questions": len(harvested_entries),
        "max_new_tokens": max_new_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entries": harvested_entries,
    }


def load_or_harvest_model_answers(
    harvested_answers_path,
    forget_split_entries,
    model_id,
    model_key,
    device,
    model_dtype,
    max_new_tokens,
):
    """
    Load harvested answers from disk or generate and save them if missing.

    Generation is kept separate from direction extraction so cached answers can
    skip the slow regeneration step on re-runs.

    Args:
        harvested_answers_path: JSON path for load-or-save of harvested answers.
        forget_split_entries: List of dicts with question and ground_truth.
        model_id: Hugging Face model id or local path.
        model_key: Optional config key from config/models.yaml.
        device: Torch device string (cuda, mps, or cpu).
        model_dtype: Torch dtype for model weights.
        max_new_tokens: Maximum tokens to generate per question when harvesting.

    Returns:
        Dict harvest record with metadata and harvested answer entries.
    """
    harvested_answers_path = Path(harvested_answers_path)
    if harvested_answers_path.is_file():
        print(f"Loading harvested answers from {harvested_answers_path}...", flush=True)
        return load_harvested_answers(harvested_answers_path)

    print(f"Harvesting model answers (will save to {harvested_answers_path})...", flush=True)
    print(f"Loading model on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=model_dtype
    ).to(device)
    model.eval()
    print("Model loaded.", flush=True)

    harvest_record = harvest_model_answers(
        model,
        tokenizer,
        forget_split_entries,
        device,
        max_new_tokens,
        model_id,
        model_key,
    )
    save_harvested_answers(harvested_answers_path, harvest_record)
    print(f"Saved harvested answers to {harvested_answers_path}", flush=True)
    return harvest_record


def answers_differ(model_answer, ground_truth_answer):
    """
    Return True when the model answer is not equal to ground truth after normalisation.

    Args:
        model_answer: Generated answer text from the model.
        ground_truth_answer: Reference answer text from the dataset.

    Returns:
        True if the answers differ case-insensitively after stripping whitespace.
    """
    return model_answer.strip().lower() != ground_truth_answer.strip().lower()


def filter_confabulation_entries(harvested_entries):
    """
    Keep only entries where the model answer genuinely differs from ground truth.

    Args:
        harvested_entries: List of harvested answer dicts.

    Returns:
        List of entries that qualify as confabulations.
    """
    return [
        entry
        for entry in harvested_entries
        if answers_differ(entry["model_answer"], entry["ground_truth"])
    ]


def build_answer_sequence_token_ids(tokenizer, question, answer_text):
    """
    Build the full prefix-plus-answer token sequence used during generation.

    Uses the chat-template generation prompt as prefix, then appends tokenised answer
    text without special tokens so activations match on-distribution generation.

    Args:
        tokenizer: Loaded tokenizer with chat template.
        question: User question text from the TOFU dataset.
        answer_text: Answer text to append after the generation prompt.

    Returns:
        Token ids tensor of shape (1, sequence_length).
    """
    messages = [{"role": "user", "content": question}]
    prefix_token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if not isinstance(prefix_token_ids, torch.Tensor):
        prefix_token_ids = prefix_token_ids["input_ids"]

    answer_encoding = tokenizer(
        answer_text, add_special_tokens=False, return_tensors="pt"
    )
    answer_token_ids = answer_encoding["input_ids"]
    return torch.cat([prefix_token_ids, answer_token_ids], dim=-1)


def collect_last_token_activation_sums(model, tokenizer, kept_entries, device, answer_field):
    """
    Accumulate last-token residual-stream activations for full answer sequences.

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Matching tokenizer with chat template.
        kept_entries: Confabulation entries with question and answer text fields.
        device: Torch device string (cuda, mps, or cpu).
        answer_field: Entry key for answer text ('model_answer' or 'ground_truth').

    Returns:
        Tuple of (activation sums per layer, number of sequences processed).
    """
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    activation_sums = [torch.zeros(hidden_size) for _ in range(num_layers)]
    sequence_count = 0

    for entry in kept_entries:
        full_sequence_token_ids = build_answer_sequence_token_ids(
            tokenizer, entry["question"], entry[answer_field]
        ).to(device)

        with torch.no_grad():
            outputs = model(full_sequence_token_ids, output_hidden_states=True)

        for layer_index in range(num_layers):
            last_token_activation = (
                outputs.hidden_states[layer_index + 1][0, -1, :].float().cpu()
            )
            activation_sums[layer_index] += last_token_activation
        sequence_count += 1

    return activation_sums, sequence_count


def compute_mean_activations(activation_sums, sequence_count):
    """
    Convert per-layer activation sums into per-layer mean vectors.

    Args:
        activation_sums: List of summed activation tensors, one per layer.
        sequence_count: Number of sequences included in the sums.

    Returns:
        List of mean activation tensors, one per layer.
    """
    return [
        activation_sums[layer_index] / sequence_count
        for layer_index in range(len(activation_sums))
    ]


def compute_difference_in_means_directions(confabulation_means, correct_means):
    """
    Subtract correct mean activations from confabulation mean activations per layer.

    Args:
        confabulation_means: Mean last-token activations on model-answer sequences.
        correct_means: Mean last-token activations on ground-truth sequences.

    Returns:
        List of raw difference-in-means direction vectors, one per layer.
    """
    return [
        confabulation_means[layer_index] - correct_means[layer_index]
        for layer_index in range(len(confabulation_means))
    ]


def extract_confabulation_directions(
    model_id,
    num_questions,
    max_new_tokens,
    output_path,
    harvested_answers_path,
    model_key=None,
):
    """
    Harvest confabulations, contrast answer-token activations, and save directions.

    Args:
        model_id: Hugging Face model id or local path.
        num_questions: Number of forget-split questions to consider.
        max_new_tokens: Maximum tokens to generate when harvesting answers.
        output_path: Path to write the saved directions file.
        harvested_answers_path: JSON path for cached harvested model answers.
        model_key: Optional config key from config/models.yaml.

    Returns:
        Dict of metadata and per-layer direction tensors written to disk.
    """
    device, model_dtype = resolve_device_and_dtype()

    print(f"Loading TOFU '{FORGET_SPLIT}'...", flush=True)
    forget_split_entries = load_forget_split_entries(num_questions)
    print(f"Loaded {len(forget_split_entries)} questions.", flush=True)

    harvest_record = load_or_harvest_model_answers(
        harvested_answers_path=harvested_answers_path,
        forget_split_entries=forget_split_entries,
        model_id=model_id,
        model_key=model_key,
        device=device,
        model_dtype=model_dtype,
        max_new_tokens=max_new_tokens,
    )
    harvested_entries = harvest_record["entries"]

    kept_entries = filter_confabulation_entries(harvested_entries)
    print(
        f"Kept {len(kept_entries)} of {len(harvested_entries)} questions as confabulations.",
        flush=True,
    )
    if len(kept_entries) == 0:
        raise ValueError(
            "No confabulations found after filtering; cannot compute directions."
        )

    print(f"Loading model on {device} for activation extraction...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=model_dtype
    ).to(device)
    model.eval()
    print("Model loaded.", flush=True)

    print("Collecting confabulation-side activations...", flush=True)
    confabulation_progress = tqdm(
        kept_entries,
        desc="Confabulation",
        unit="sequence",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    confabulation_sums, confabulation_count = collect_last_token_activation_sums(
        model,
        tokenizer,
        confabulation_progress,
        device,
        answer_field="model_answer",
    )
    confabulation_means = compute_mean_activations(
        confabulation_sums, confabulation_count
    )

    print("Collecting correct-side activations...", flush=True)
    correct_progress = tqdm(
        kept_entries,
        desc="Correct",
        unit="sequence",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    correct_sums, correct_count = collect_last_token_activation_sums(
        model,
        tokenizer,
        correct_progress,
        device,
        answer_field="ground_truth",
    )
    correct_means = compute_mean_activations(correct_sums, correct_count)

    directions = compute_difference_in_means_directions(
        confabulation_means, correct_means
    )

    save_record = {
        "model": model_id,
        "model_key": model_key,
        "split": FORGET_SPLIT,
        "num_questions_requested": num_questions,
        "num_questions_harvested": len(harvested_entries),
        "num_confabulations_kept": len(kept_entries),
        "max_new_tokens": max_new_tokens,
        "harvested_answers_file": str(harvested_answers_path),
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "directions": directions,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_record, output_path)
    print(f"Saved confabulation directions to {output_path}", flush=True)

    return save_record


def main():
    """Parse CLI arguments and extract confabulation directions."""
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
        help="number of forget-split questions to harvest and consider",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="maximum tokens to generate when harvesting model answers",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to save per-layer direction vectors (default from config)",
    )
    parser.add_argument(
        "--harvested-answers",
        default=None,
        help="path to harvested-answers JSON (loads if present; saves after harvest)",
    )
    arguments = parser.parse_args()

    model_entry = get_model(arguments.model_key)
    output_path = arguments.output or model_entry["outputs"]["confabulation_direction"]
    harvested_answers_path = arguments.harvested_answers or default_harvested_answers_path(
        output_path
    )

    extract_confabulation_directions(
        model_id=model_entry["hf_id"],
        num_questions=arguments.num_questions,
        max_new_tokens=arguments.max_new_tokens,
        output_path=output_path,
        harvested_answers_path=harvested_answers_path,
        model_key=arguments.model_key,
    )


if __name__ == "__main__":
    main()
