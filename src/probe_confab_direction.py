#!/usr/bin/env python3
"""
probe_confab_direction.py - train per-layer linear probes on confab vs correct activations.

Loads harvested forget answers (same cache as confabulation_direction.py), keeps only
questions where the model confabulates, and fits a logistic regression probe per layer
on last-token activations from full prefix-plus-answer sequences. Use the resulting
per-layer test accuracy to choose steering layers for confabulation directions.

Setup:
    pip install -r requirements.txt
    pip install scikit-learn

Usage:
    python src/probe_confab_direction.py --model-key npo_unlearned
"""
import argparse

from model_config import get_model
from probing.linear_probe import train_confab_linear_probes
from utils.paths import (
    default_harvested_answers_path,
    default_probe_confab_direction_output_path,
)


def main():
    """Parse CLI arguments and train confab-vs-correct linear probes."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        required=True,
        help="short name from config/models.yaml (e.g. npo_unlearned)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to write per-layer probe accuracy JSON (default from config)",
    )
    parser.add_argument(
        "--harvested-answers",
        default=None,
        help="path to harvested-answers JSON (must exist; from confabulation_direction)",
    )
    arguments = parser.parse_args()

    model_entry = get_model(arguments.model_key)
    confabulation_direction_path = model_entry["outputs"]["confabulation_direction"]
    harvested_answers_path = arguments.harvested_answers or default_harvested_answers_path(
        confabulation_direction_path
    )
    output_path = arguments.output or default_probe_confab_direction_output_path(
        model_entry, arguments.model_key
    )

    train_confab_linear_probes(
        model_id=model_entry["hf_id"],
        harvested_answers_path=harvested_answers_path,
        output_path=output_path,
        model_key=arguments.model_key,
    )


if __name__ == "__main__":
    main()
