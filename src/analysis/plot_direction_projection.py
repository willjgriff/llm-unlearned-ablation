"""Plot forget vs retain direction projection kernel density estimates."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from utils.constants import HARMFUL_SPLIT, HARMLESS_SPLIT

DEFAULT_PROJECTION_PLOT_LAYER = 14


def _get_layer_projections_from_record(saved_record, layer):
    """
    Read forget and retain direction projections for one layer from a save record.

    Args:
        saved_record: Dict written by extract_refusal_directions.
        layer: Transformer layer index to read projections for.

    Returns:
        Tuple of (forget projection values, retain projection values, model label).

    Raises:
        KeyError: If the record has no saved direction_projections.
        ValueError: If the layer index is out of range.
    """
    direction_projections = saved_record.get("direction_projections")
    if direction_projections is None:
        raise KeyError(
            "No direction_projections found in save record. "
            "Re-run refusal_direction.py to regenerate the directions file."
        )

    num_layers = saved_record["num_layers"]
    if layer < 0 or layer >= num_layers:
        raise ValueError(
            f"Layer {layer} is out of range for {num_layers} layers."
        )

    model_label = saved_record.get("model_key") or saved_record.get("model", "unknown")

    return (
        direction_projections["forget"][layer],
        direction_projections["retain"][layer],
        model_label,
    )


def load_layer_direction_projections(directions_file, layer):
    """
    Load per-question direction projections for forget and retain at one layer.

    Args:
        directions_file: Path to a saved refusal directions .pt file.
        layer: Transformer layer index to read projections for.

    Returns:
        Tuple of (forget projection values, retain projection values, model label).

    Raises:
        KeyError: If the directions file has no saved direction_projections.
        ValueError: If the layer index is out of range.
    """
    saved_record = torch.load(directions_file, map_location="cpu", weights_only=False)
    return _get_layer_projections_from_record(saved_record, layer)


def build_kernel_density_curve(values, evaluation_points):
    """
    Evaluate a Gaussian kernel density estimate over a grid of points.

    Args:
        values: One-dimensional array of scalar projection values.
        evaluation_points: Points at which to evaluate the KDE.

    Returns:
        One-dimensional array of KDE values aligned with evaluation_points.
    """
    if len(values) < 2:
        raise ValueError("At least two projection values are required for a KDE plot.")

    unique_values = np.unique(values)
    if len(unique_values) == 1:
        density_curve = np.zeros_like(evaluation_points, dtype=float)
        nearest_index = np.argmin(np.abs(evaluation_points - unique_values[0]))
        density_curve[nearest_index] = 1.0
        return density_curve

    kernel_density = gaussian_kde(values)
    return kernel_density(evaluation_points)


def plot_direction_projection_kde(
    forget_projections,
    retain_projections,
    layer,
    output_path,
    model_label,
    harmful_split=HARMFUL_SPLIT,
    harmless_split=HARMLESS_SPLIT,
):
    """
    Plot overlapping KDE curves for forget and retain direction projections.

    Args:
        forget_projections: Per-question dot products for the forget split.
        retain_projections: Per-question dot products for the retain split.
        layer: Transformer layer index shown in the plot title.
        output_path: Path to write the PNG plot.
        model_label: Model key or Hugging Face id shown on the plot.
        harmful_split: Forget split label used in the legend.
        harmless_split: Retain split label used in the legend.

    Returns:
        Path string for the saved plot file.
    """
    forget_values = np.array(forget_projections, dtype=float)
    retain_values = np.array(retain_projections, dtype=float)
    combined_values = np.concatenate([forget_values, retain_values])

    value_padding = max((combined_values.max() - combined_values.min()) * 0.05, 1e-6)
    evaluation_min = combined_values.min() - value_padding
    evaluation_max = combined_values.max() + value_padding
    evaluation_points = np.linspace(evaluation_min, evaluation_max, 400)

    forget_density = build_kernel_density_curve(forget_values, evaluation_points)
    retain_density = build_kernel_density_curve(retain_values, evaluation_points)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        evaluation_points,
        forget_density,
        label=harmful_split,
        color="#d62728",
        linewidth=2,
    )
    axis.plot(
        evaluation_points,
        retain_density,
        label=harmless_split,
        color="#1f77b4",
        linewidth=2,
    )
    axis.set_title(f"Direction projection KDE (layer {layer})")
    axis.set_xlabel("Activation dot product with direction vector")
    axis.set_ylabel("Density")
    axis.legend()
    figure.text(0.5, 0.98, model_label, ha="center", va="top", fontsize=10)
    figure.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved direction projection plot to {output_path}", flush=True)
    return str(output_path)


def plot_direction_projection_from_record(save_record, layer, output_path):
    """
    Plot direction projection KDEs from an in-memory refusal directions save record.

    Args:
        save_record: Dict returned by extract_refusal_directions.
        layer: Transformer layer index to plot.
        output_path: Path to write the PNG plot.

    Returns:
        Path string for the saved plot file.
    """
    forget_projections, retain_projections, model_label = (
        _get_layer_projections_from_record(save_record, layer)
    )
    return plot_direction_projection_kde(
        forget_projections,
        retain_projections,
        layer,
        output_path,
        model_label,
    )


def plot_refusal_direction_projection(
    directions_file,
    layer,
    output_path,
):
    """
    Load saved refusal direction projections and write a KDE plot for one layer.

    Args:
        directions_file: Path to a saved refusal directions .pt file.
        layer: Transformer layer index to plot.
        output_path: Path to write the PNG plot.

    Returns:
        Path string for the saved plot file.
    """
    forget_projections, retain_projections, model_label = (
        load_layer_direction_projections(directions_file, layer)
    )
    return plot_direction_projection_kde(
        forget_projections,
        retain_projections,
        layer,
        output_path,
        model_label,
    )
