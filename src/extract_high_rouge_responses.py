#!/usr/bin/env python3
"""
extract_high_rouge_responses.py - extract high-ROUGE answers for manual review.

Reads probe or ablate-and-probe JSON files and writes two sections for manual
verification: above_0.6 (rouge_l > 0.6) and above_0.3 (0.3 < rouge_l <= 0.6).
Sweep entries produce one row per steering coefficient. Each result puts
model_answer last, immediately before rouge_l.

Usage:
    python src/extract_high_rouge_responses.py \\
        results/probe/idk_nll_unlearned_lr3e-05_alpha10_epoch5.json

    python src/extract_high_rouge_responses.py \\
        results/ablate-and-probe/idk_nll_unlearned_lr3e-05_alpha10_epoch5/negsteer_layer14_coef2.5_refusal.json

    python src/extract_high_rouge_responses.py --input-dir results/probe
"""
import argparse

from analysis.extract_high_rouge import (
    extract_high_rouge_responses_batch,
    resolve_input_paths,
)


def main():
    """Parse CLI arguments and extract high-ROUGE responses."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="*",
        help="one or more probe or ablate-and-probe JSON files",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="directory of probe JSON files to process (skips *_high_rouge.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to write filtered JSON when processing a single input file",
    )
    arguments = parser.parse_args()

    input_paths = resolve_input_paths(
        input_paths=arguments.input or None,
        input_dir=arguments.input_dir,
    )
    extract_high_rouge_responses_batch(input_paths, output_path=arguments.output)


if __name__ == "__main__":
    main()
