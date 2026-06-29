#!/usr/bin/env python3
"""
extract_ablate_probe_coefficients.py - filter and flatten ablate-and-probe JSON by coefficient.

Reads an ablate-and-probe results file (single-coefficient or sweep format), keeps
only the requested steering coefficients, and writes output in the same top-level
schema with per-question answers collapsed into model_answers_coefficient{value} keys.

Usage:
    python src/extract_ablate_probe_coefficients.py \\
        results/ablate-and-probe/idk_nll_unlearned_lr3e-05_alpha10_epoch5/negsteer_sweep_layer12.json \\
        --coefficients 1.5 2.0 5.0 \\
        --output results/ablate-and-probe/filtered_layer12.json
"""
import argparse

from analysis.extract_coefficients import (
    default_output_path,
    extract_ablate_probe_coefficients,
)


def main():
    """Parse CLI arguments and extract selected steering coefficient answers."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="path to an ablate-and-probe JSON file",
    )
    parser.add_argument(
        "--coefficients",
        nargs="+",
        type=float,
        required=True,
        help="steering coefficients to keep (space-separated floats)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to write filtered JSON (default: input stem + _filtered.json)",
    )
    arguments = parser.parse_args()

    output_path = arguments.output or default_output_path(arguments.input)

    extract_ablate_probe_coefficients(
        input_path=arguments.input,
        requested_coefficients=arguments.coefficients,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
