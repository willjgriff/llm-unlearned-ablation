"""Confabulation direction extraction via difference-in-means."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.activations import (
    collect_last_token_activation_sums_for_sequences,
    compute_difference_in_means_directions,
    compute_mean_activations,
)
from utils.constants import FORGET_SPLIT
from utils.device import resolve_device_and_dtype
from utils.inference import generate_answer
from utils.tofu_data import load_forget_split_entries


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
    forget_split_entries = load_forget_split_entries(num_questions, FORGET_SPLIT)
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
    confabulation_sums, confabulation_count = (
        collect_last_token_activation_sums_for_sequences(
            model,
            tokenizer,
            confabulation_progress,
            device,
            answer_field="model_answer",
        )
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
    correct_sums, correct_count = collect_last_token_activation_sums_for_sequences(
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
