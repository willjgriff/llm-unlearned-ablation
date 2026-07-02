#!/usr/bin/env python3
"""
plot_direction_projection.py - plot forget vs retain direction projection KDEs.

Loads per-question dot products saved in a refusal directions .pt file and plots
overlapping kernel density estimate curves for forget10 and retain90 at one layer.

Setup:
    pip install -r requirements.txt

Usage:
    python src/plot_direction_projection.py --model-key npo_unlearned --layer 14
"""
import argparse

from analysis.plot_direction_projection import plot_refusal_direction_projection
from model_config import get_model
from utils.paths import default_refusal_direction_projection_plot_path


def main():
    """Parse CLI arguments and plot direction projection KDEs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        required=True,
        help="short name from config/models.yaml (e.g. npo_unlearned)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="transformer layer index to plot",
    )
    parser.add_argument(
        "--directions-file",
        default=None,
        help="path to directions .pt file (default from config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to write PNG plot (default from config)",
    )
    arguments = parser.parse_args()

    model_entry = get_model(arguments.model_key)
    directions_file = (
        arguments.directions_file or model_entry["outputs"]["refusal_direction"]
    )
    output_path = arguments.output or default_refusal_direction_projection_plot_path(
        model_entry, arguments.model_key, arguments.layer
    )

    plot_refusal_direction_projection(
        directions_file=directions_file,
        layer=arguments.layer,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
