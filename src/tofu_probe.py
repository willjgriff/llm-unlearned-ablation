#!/usr/bin/env python3
"""
tofu_probe.py - prompt a TOFU (un)learned checkpoint with forget-set author
questions and save model answers alongside TOFU ground-truth to JSON.

Setup:
    pip install -r requirements.txt

Usage (run once per model to compare behavior):
    python src/tofu_probe.py --model-key baseline_full
"""
import argparse

from model_config import get_model
from probing.behavioral import run_probe


def main():
    """Parse CLI arguments and run the TOFU forget-set probe."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        required=True,
        help="short name from config/models.yaml (e.g. baseline_full, npo_unlearned)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="number of questions to probe",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--output",
        default=None,
        help="path to write JSON results (default from config; pass empty string to skip)",
    )
    arguments = parser.parse_args()

    model_entry = get_model(arguments.model_key)
    output_path = arguments.output
    if output_path is None:
        output_path = model_entry["outputs"]["probe"]
    elif output_path == "":
        output_path = None

    run_probe(
        model_id=model_entry["hf_id"],
        num_questions=arguments.num_questions,
        max_new_tokens=arguments.max_new_tokens,
        output_path=output_path,
        model_key=arguments.model_key,
    )


if __name__ == "__main__":
    main()
