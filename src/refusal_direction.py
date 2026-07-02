#!/usr/bin/env python3
"""
refusal_direction.py - extract per-layer refusal directions via difference-in-means.

Uses TOFU forget10 questions as the "harmful" side (prompts the model deflects on)
and retain90 as the "harmless" side (prompts the model answers normally). Saves raw
difference-in-means direction vectors and per-question direction projections for
every layer to disk, and saves a direction projection KDE plot for layer 14.

Setup:
    pip install -r requirements.txt

Usage:
    python src/refusal_direction.py --model-key npo_unlearned
"""
import argparse

from direction_calculation.refusal import extract_refusal_directions
from model_config import get_model
from analysis.plot_direction_projection import DEFAULT_PROJECTION_PLOT_LAYER
from utils.paths import default_refusal_direction_projection_plot_path


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
    parser.add_argument(
        "--projection-plot-layer",
        type=int,
        default=DEFAULT_PROJECTION_PLOT_LAYER,
        help="layer index for the auto-generated projection KDE plot (default: 14)",
    )
    parser.add_argument(
        "--projection-plot-output",
        default=None,
        help="path to write the projection KDE PNG (default from config)",
    )
    arguments = parser.parse_args()

    model_entry = get_model(arguments.model_key)
    output_path = arguments.output or model_entry["outputs"]["refusal_direction"]
    projection_plot_output_path = (
        arguments.projection_plot_output
        or default_refusal_direction_projection_plot_path(
            model_entry, arguments.model_key, arguments.projection_plot_layer
        )
    )

    extract_refusal_directions(
        model_id=model_entry["hf_id"],
        num_questions=arguments.num_questions,
        output_path=output_path,
        model_key=arguments.model_key,
        projection_plot_layer=arguments.projection_plot_layer,
        projection_plot_output_path=projection_plot_output_path,
    )


if __name__ == "__main__":
    main()
