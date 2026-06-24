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

from model_config import get_model
from probing.linear_probe import train_linear_probes
from utils.paths import default_probe_direction_output_path


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
    output_path = arguments.output or default_probe_direction_output_path(
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
