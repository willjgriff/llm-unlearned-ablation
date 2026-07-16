#!/usr/bin/env python3
"""
tofu_probe.py - prompt a TOFU (un)learned checkpoint with forget-set author
questions and save model answers alongside TOFU ground-truth to JSON.

Setup:
    pip install -r requirements.txt

Usage (run once per model to compare behavior):
    python src/tofu_probe.py --model-key baseline_full
    python src/tofu_probe.py --model-key baseline_full --question-mode perturbed
"""
import argparse
from pathlib import Path

from model_config import get_model
from probing.behavioral import run_probe
from utils.constants import QUESTION_MODE_ORIGINAL, QUESTION_MODE_PERTURBED


def resolve_probe_output_path(default_path, question_mode, output_override=None):
    """
    Resolve the JSON output path for a probe run.

    Args:
        default_path: Default probe output path from config/models.yaml.
        question_mode: Either original or perturbed.
        output_override: Explicit --output value, or None to use the default.

    Returns:
        Output path string, or None when writing should be skipped.
    """
    if output_override == "":
        return None
    if output_override is not None:
        return output_override

    output_path = Path(default_path)
    if question_mode == QUESTION_MODE_PERTURBED:
        output_path = output_path.with_name(
            f"{output_path.stem}_perturbed{output_path.suffix}"
        )
    return str(output_path)


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
    parser.add_argument(
        "--question-mode",
        choices=[QUESTION_MODE_ORIGINAL, QUESTION_MODE_PERTURBED],
        default=QUESTION_MODE_ORIGINAL,
        help="use original forget10 questions or paraphrased forget10_perturbed prompts",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="optional TOFU config override (default: forget10 or forget10_perturbed)",
    )
    arguments = parser.parse_args()

    model_entry = get_model(arguments.model_key)
    output_path = resolve_probe_output_path(
        model_entry["outputs"]["probe"],
        arguments.question_mode,
        arguments.output,
    )

    run_probe(
        model_id=model_entry["hf_id"],
        num_questions=arguments.num_questions,
        max_new_tokens=arguments.max_new_tokens,
        output_path=output_path,
        model_key=arguments.model_key,
        question_mode=arguments.question_mode,
        split_name=arguments.split,
    )


if __name__ == "__main__":
    main()
