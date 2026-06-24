#!/usr/bin/env python3
"""
refusal_direction.py - extract per-layer refusal directions via difference-in-means.

Uses TOFU forget10 questions as the "harmful" side (prompts the model deflects on)
and retain90 as the "harmless" side (prompts the model answers normally). Saves raw
difference-in-means direction vectors for every layer to disk.

Setup:
    pip install -r requirements.txt

Usage:
    python src/refusal_direction.py --model-key npo_unlearned
"""
import argparse

from direction_calculation.refusal import extract_refusal_directions
from model_config import get_model


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
