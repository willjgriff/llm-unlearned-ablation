#!/usr/bin/env python3
"""
plot_rouge_comparison.py - grouped bar chart of ROUGE-L recovery counts.

Reads count_above_0.3 or count_above_0.6 from probe and ablate-and-probe result
files and plots recovery counts for IDK-NLL, NPO, and baseline models with
unsteered/steered bars.

Setup:
    pip install -r requirements.txt

Usage:
    python src/plot_rouge_comparison.py \\
        --idk-nll-unsteered results/probe/idk_nll_unlearned_lr3e-05_alpha10_epoch5.json \\
        --idk-nll-steered results/ablate-and-probe/idk_nll_unlearned_lr3e-05_alpha10_epoch5/negsteer_layer14_coef2.5_refusal.json \\
        --npo-unsteered results/probe/npo_unlearned_lr2e-05_beta0.5_alpha5_epoch5.json \\
        --npo-steered results/ablate-and-probe/npo_unlearned_lr2e-05_beta0.5_alpha5_epoch5/negsteer_layer14_coef1_refusal.json \\
        --baseline results/probe/baseline_full.json
"""
import argparse

from analysis.plot_rouge_comparison import (
    default_rouge_comparison_plot_path,
    plot_rouge_comparison,
)


def main():
    """Parse CLI arguments and plot ROUGE-L recovery comparison bars."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--idk-nll-unsteered",
        required=True,
        help="path to unsteered IDK-NLL probe JSON",
    )
    parser.add_argument(
        "--idk-nll-steered",
        required=True,
        help="path to steered IDK-NLL ablate-and-probe JSON",
    )
    parser.add_argument(
        "--npo-unsteered",
        required=True,
        help="path to unsteered NPO probe JSON",
    )
    parser.add_argument(
        "--npo-steered",
        required=True,
        help="path to steered NPO ablate-and-probe JSON",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="path to baseline probe JSON",
    )
    parser.add_argument(
        "--threshold",
        choices=["0.3", "0.6"],
        default="0.3",
        help="ROUGE-L threshold to plot: above 0.3 or above 0.6",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to write PNG plot (default: results/rouge-comparison/rouge_comparison_{threshold}.png)",
    )
    arguments = parser.parse_args()

    output_path = arguments.output or default_rouge_comparison_plot_path(
        arguments.threshold
    )
    plot_rouge_comparison(
        idk_nll_unsteered_path=arguments.idk_nll_unsteered,
        idk_nll_steered_path=arguments.idk_nll_steered,
        npo_unsteered_path=arguments.npo_unsteered,
        npo_steered_path=arguments.npo_steered,
        baseline_path=arguments.baseline,
        output_path=output_path,
        threshold=arguments.threshold,
    )


if __name__ == "__main__":
    main()
