#!/usr/bin/env python3
"""
probe_direction.py - train per-layer linear probes on residual stream activations.

Forwards forget10 and retain90 TOFU questions through the model, extracts last-token
residual stream activations at every layer, and fits a logistic regression probe per
layer to classify forget vs retain. Saves per-layer train and test accuracy to JSON.

Setup:
    pip install -r requirements.txt
    pip install scikit-learn

Usage:
    python src/probe_direction.py --model-key npo_unlearned
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_config import get_model

FORGET_SPLIT = "forget10"
RETAIN_SPLIT = "retain90"
TRAIN_FRACTION = 0.8
RANDOM_STATE = 42


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


def collect_last_token_activations(model, tokenizer, questions, device):
    """
    Collect last-token residual-stream activations at every layer for each question.

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Matching tokenizer with chat template.
        questions: Iterable of question strings.
        device: Torch device string (cuda, mps, or cpu).

    Returns:
        Tuple of (per-layer activation lists, number of questions processed). Each
        layer list contains one hidden-state vector per question.
    """
    num_layers = model.config.num_hidden_layers
    per_layer_activations = [[] for _ in range(num_layers)]
    question_count = 0

    for question in questions:
        input_token_ids = tokenize_question(tokenizer, question, device)

        with torch.no_grad():
            outputs = model(input_token_ids, output_hidden_states=True)

        for layer_index in range(num_layers):
            last_token_activation = (
                outputs.hidden_states[layer_index + 1][0, -1, :].float().cpu().numpy()
            )
            per_layer_activations[layer_index].append(last_token_activation)
        question_count += 1

    return per_layer_activations, question_count


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


def default_output_path(model_entry, model_key):
    """
    Derive the default linear-probe output path for a model.

    Uses outputs.probe_direction from config when present, otherwise falls back to
    results/probe-direction/{model_key}.json.

    Args:
        model_entry: Model dict from config/models.yaml.
        model_key: Config key for the model.

    Returns:
        Default output path string.
    """
    configured_output = model_entry.get("outputs", {}).get("probe_direction")
    if configured_output:
        return configured_output
    return f"results/probe-direction/{model_key}.json"


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
    device, model_dtype = resolve_device_and_dtype()

    print(f"Loading model on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=model_dtype
    ).to(device)
    model.eval()
    print("Model loaded.", flush=True)

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


def main():
    """Parse CLI arguments and train linear direction probes."""
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
        help="path to write per-layer probe accuracy JSON (default from config)",
    )
    arguments = parser.parse_args()

    model_entry = get_model(arguments.model_key)
    output_path = arguments.output or default_output_path(
        model_entry, arguments.model_key
    )

    train_linear_probes(
        model_id=model_entry["hf_id"],
        num_questions=arguments.num_questions,
        output_path=output_path,
        model_key=arguments.model_key,
    )


if __name__ == "__main__":
    main()
