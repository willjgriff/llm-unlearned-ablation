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

from direction_calculation.confabulation import extract_confabulation_directions
from model_config import get_model
from utils.paths import default_harvested_answers_path


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
