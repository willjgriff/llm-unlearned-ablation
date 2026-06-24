"""Per-layer linear probe training on forget vs retain activations."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.activations import collect_last_token_activations
from utils.constants import FORGET_SPLIT, RANDOM_STATE, RETAIN_SPLIT, TRAIN_FRACTION
from utils.model_loading import load_model_and_tokenizer
from utils.tofu_data import load_questions


def train_layer_probe(forget_activations, retain_activations):
    """
    Fit a logistic regression probe on forget vs retain activations for one layer.

    Args:
        forget_activations: List of activation vectors from the forget split.
        retain_activations: List of activation vectors from the retain split.

    Returns:
        Dict with train_accuracy and test_accuracy on an 80/20 stratified split.
    """
    forget_features = np.array(forget_activations)
    retain_features = np.array(retain_activations)
    feature_matrix = np.vstack([forget_features, retain_features])
    labels = np.array([1] * len(forget_features) + [0] * len(retain_features))

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


def train_linear_probes(model_id, num_questions, output_path, model_key=None):
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
        "model": model_id,
        "model_key": model_key,
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
    print(f"Saved probe results to {output_path}", flush=True)

    return probe_record
