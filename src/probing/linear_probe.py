"""Per-layer linear probe training on forget vs retain or confab vs correct activations."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from direction_calculation.confabulation import (
    filter_confabulation_entries,
    load_harvested_answers,
)
from utils.activations import (
    collect_last_token_activations,
    collect_last_token_activations_for_sequences,
)
from utils.constants import FORGET_SPLIT, RANDOM_STATE, RETAIN_SPLIT, TRAIN_FRACTION
from utils.model_loading import load_model_and_tokenizer
from utils.tofu_data import load_questions


def train_layer_probe(positive_activations, negative_activations):
    """
    Fit a logistic regression probe on paired positive vs negative activations.

    Args:
        positive_activations: List of activation vectors for the positive class.
        negative_activations: List of activation vectors for the negative class.

    Returns:
        Dict with train_accuracy and test_accuracy on an 80/20 stratified split.
    """
    positive_features = np.array(positive_activations)
    negative_features = np.array(negative_activations)
    feature_matrix = np.vstack([positive_features, negative_features])
    labels = np.array([1] * len(positive_features) + [0] * len(negative_features))

    train_features, test_features, train_labels, test_labels = train_test_split(
        feature_matrix,
        labels,
        test_size=1.0 - TRAIN_FRACTION,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    probe = LogisticRegression(max_iter=1000)
    probe.fit(train_features, train_labels)

    train_predictions = probe.predict(train_features)
    test_predictions = probe.predict(test_features)

    return {
        "train_accuracy": float(accuracy_score(train_labels, train_predictions)),
        "test_accuracy": float(accuracy_score(test_labels, test_predictions)),
    }


def train_refusal_linear_probes(model_id, num_questions, output_path, model_key=None):
    """
    Train and evaluate per-layer linear probes for forget10 vs retain90 activations.

    Args:
        model_id: Hugging Face model id or local path.
        num_questions: Number of questions to use from each TOFU split.
        output_path: Path to write per-layer accuracy JSON.
        model_key: Optional config key from config/models.yaml.

    Returns:
        Dict of metadata and per-layer probe accuracy results.
    """
    model, tokenizer, device, _model_dtype = load_model_and_tokenizer(model_id)

    print(f"Loading TOFU '{FORGET_SPLIT}' and '{RETAIN_SPLIT}'...", flush=True)
    forget_questions = load_questions(FORGET_SPLIT, num_questions)
    retain_questions = load_questions(RETAIN_SPLIT, num_questions)
    print(
        f"Using {len(forget_questions)} forget and {len(retain_questions)} "
        f"retain questions.",
        flush=True,
    )

    print("Collecting forget activations...", flush=True)
    forget_progress = tqdm(
        forget_questions,
        desc="Forget",
        unit="question",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    forget_activations, forget_count = collect_last_token_activations(
        model, tokenizer, forget_progress, device
    )

    print("Collecting retain activations...", flush=True)
    retain_progress = tqdm(
        retain_questions,
        desc="Retain",
        unit="question",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    retain_activations, retain_count = collect_last_token_activations(
        model, tokenizer, retain_progress, device
    )

    num_layers = model.config.num_hidden_layers
    layer_results = []
    print("Training per-layer probes...", flush=True)
    for layer_index in range(num_layers):
        layer_probe_scores = train_layer_probe(
            forget_activations[layer_index],
            retain_activations[layer_index],
        )
        layer_result = {
            "layer": layer_index,
            "train_accuracy": layer_probe_scores["train_accuracy"],
            "test_accuracy": layer_probe_scores["test_accuracy"],
        }
        layer_results.append(layer_result)
        print(
            f"Layer {layer_index} train accuracy: "
            f"{layer_result['train_accuracy']:.4f}, "
            f"test accuracy: {layer_result['test_accuracy']:.4f}",
            flush=True,
        )

    probe_record = {
        "probe_type": "forget_vs_retain",
        "model": model_id,
        "model_key": model_key,
        "activation_protocol": "last_token_prompt",
        "positive_class": "forget",
        "negative_class": "retain",
        "forget_split": FORGET_SPLIT,
        "retain_split": RETAIN_SPLIT,
        "num_questions_per_split": num_questions,
        "forget_questions_used": forget_count,
        "retain_questions_used": retain_count,
        "num_layers": num_layers,
        "hidden_size": model.config.hidden_size,
        "device": device,
        "train_fraction": TRAIN_FRACTION,
        "random_state": RANDOM_STATE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "layer_results": layer_results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(probe_record, indent=2) + "\n")
    print(f"Saved refusal probe results to {output_path}", flush=True)

    return probe_record


def train_confab_linear_probes(
    model_id,
    harvested_answers_path,
    output_path,
    model_key=None,
):
    """
    Train and evaluate per-layer linear probes for confab vs correct activations.

    Uses the same harvested forget answers and confabulation filter as
    confabulation direction extraction, with last-token activations on full
    prefix-plus-answer sequences.

    Args:
        model_id: Hugging Face model id or local path.
        harvested_answers_path: Path to harvested-answers JSON from confabulation runs.
        output_path: Path to write per-layer accuracy JSON.
        model_key: Optional config key from config/models.yaml.

    Returns:
        Dict of metadata and per-layer probe accuracy results.
    """
    harvested_answers_path = Path(harvested_answers_path)
    if not harvested_answers_path.is_file():
        raise FileNotFoundError(
            f"Harvested answers not found at {harvested_answers_path}. "
            "Run confabulation_direction.py first."
        )

    print(f"Loading harvested answers from {harvested_answers_path}...", flush=True)
    harvest_record = load_harvested_answers(harvested_answers_path)
    harvested_entries = harvest_record["entries"]
    kept_entries = filter_confabulation_entries(harvested_entries)
    print(
        f"Kept {len(kept_entries)} of {len(harvested_entries)} questions as confabulations.",
        flush=True,
    )
    if len(kept_entries) == 0:
        raise ValueError(
            "No confabulations found after filtering; cannot train confab probes."
        )

    model, tokenizer, device, _model_dtype = load_model_and_tokenizer(model_id)

    print("Collecting confabulation-side activations...", flush=True)
    confabulation_progress = tqdm(
        kept_entries,
        desc="Confabulation",
        unit="sequence",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    confabulation_activations, confabulation_count = (
        collect_last_token_activations_for_sequences(
            model,
            tokenizer,
            confabulation_progress,
            device,
            answer_field="model_answer",
        )
    )

    print("Collecting correct-side activations...", flush=True)
    correct_progress = tqdm(
        kept_entries,
        desc="Correct",
        unit="sequence",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    correct_activations, correct_count = collect_last_token_activations_for_sequences(
        model,
        tokenizer,
        correct_progress,
        device,
        answer_field="ground_truth",
    )

    num_layers = model.config.num_hidden_layers
    layer_results = []
    print("Training per-layer probes...", flush=True)
    for layer_index in range(num_layers):
        layer_probe_scores = train_layer_probe(
            confabulation_activations[layer_index],
            correct_activations[layer_index],
        )
        layer_result = {
            "layer": layer_index,
            "train_accuracy": layer_probe_scores["train_accuracy"],
            "test_accuracy": layer_probe_scores["test_accuracy"],
        }
        layer_results.append(layer_result)
        print(
            f"Layer {layer_index} train accuracy: "
            f"{layer_result['train_accuracy']:.4f}, "
            f"test accuracy: {layer_result['test_accuracy']:.4f}",
            flush=True,
        )

    probe_record = {
        "probe_type": "confab_vs_correct",
        "model": model_id,
        "model_key": model_key,
        "split": FORGET_SPLIT,
        "activation_protocol": "last_token_full_sequence",
        "positive_class": "confabulation",
        "negative_class": "correct",
        "harvested_answers_file": str(harvested_answers_path),
        "num_questions_harvested": len(harvested_entries),
        "num_confabulations_kept": len(kept_entries),
        "confabulation_sequences_used": confabulation_count,
        "correct_sequences_used": correct_count,
        "num_layers": num_layers,
        "hidden_size": model.config.hidden_size,
        "device": device,
        "train_fraction": TRAIN_FRACTION,
        "random_state": RANDOM_STATE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "layer_results": layer_results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(probe_record, indent=2) + "\n")
    print(f"Saved confab probe results to {output_path}", flush=True)

    return probe_record
