#!/usr/bin/env python3
"""
extract_high_rouge_responses.py - extract high-ROUGE ablate-and-probe answers for review.

Reads an ablate-and-probe JSON file and writes two sections for manual verification:
above_0.6 (rouge_l > 0.6) and above_0.3 (0.3 < rouge_l <= 0.6). Sweep entries
produce one row per steering coefficient. Each result puts model_answer last,
immediately before rouge_l.

Usage:
    python src/extract_high_rouge_responses.py \\
        results/ablate-and-probe/idk_nll_unlearned_lr3e-05_alpha10_epoch5/negsteer_layer14_coef2.5.json
"""
import argparse

from analysis.extract_high_rouge import (
    default_high_rouge_output_path,
    extract_high_rouge_responses,
)


def main():
    """Parse CLI arguments and extract high-ROUGE responses."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="path to an ablate-and-probe JSON file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to write filtered JSON (default: input stem + _high_rouge.json)",
    )
    arguments = parser.parse_args()

    output_path = arguments.output or default_high_rouge_output_path(arguments.input)
    extract_high_rouge_responses(arguments.input, output_path)


if __name__ == "__main__":
    main()
