"""Refusal direction extraction via difference-in-means."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from analysis.plot_direction_projection import (
    DEFAULT_PROJECTION_PLOT_LAYER,
    plot_direction_projection_from_record,
)
from utils.activations import (
    collect_last_token_activations,
    compute_difference_in_means_directions,
    compute_mean_activations_from_lists,
    compute_per_question_direction_projections,
)
from utils.constants import HARMFUL_SPLIT, HARMLESS_SPLIT
from utils.model_loading import load_model_and_tokenizer
from utils.paths import refusal_direction_projection_plot_path
from utils.tofu_data import load_questions


def extract_refusal_directions(
    model_id,
    num_questions,
    output_path,
    model_key=None,
    projection_plot_layer=DEFAULT_PROJECTION_PLOT_LAYER,
    projection_plot_output_path=None,
):
    """
    Extract and save per-layer refusal directions for a model checkpoint.

    Also saves per-question dot products of activations with each layer direction
    for both forget and retain splits.

    Args:
        model_id: Hugging Face model id or local path.
        num_questions: Number of questions to use from each TOFU split.
        output_path: Path to write the saved directions file.
        model_key: Optional config key from config/models.yaml.
        projection_plot_layer: Layer index for the auto-generated projection plot.
        projection_plot_output_path: Optional PNG output path for the projection plot.

    Returns:
        Dict of metadata and per-layer direction tensors written to disk.
    """
    model, tokenizer, device, _model_dtype = load_model_and_tokenizer(model_id)

    print(f"Loading TOFU '{HARMFUL_SPLIT}' and '{HARMLESS_SPLIT}'...", flush=True)
    harmful_questions = load_questions(HARMFUL_SPLIT, num_questions)
    harmless_questions = load_questions(HARMLESS_SPLIT, num_questions)
    print(
        f"Using {len(harmful_questions)} harmful and {len(harmless_questions)} "
        f"harmless questions.",
        flush=True,
    )

    print("Collecting harmful activations...", flush=True)
    harmful_progress = tqdm(
        harmful_questions,
        desc="Harmful",
        unit="question",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    harmful_activations, harmful_count = collect_last_token_activations(
        model, tokenizer, harmful_progress, device
    )
    harmful_means = compute_mean_activations_from_lists(harmful_activations)

    print("Collecting harmless activations...", flush=True)
    harmless_progress = tqdm(
        harmless_questions,
        desc="Harmless",
        unit="question",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    harmless_activations, harmless_count = collect_last_token_activations(
        model, tokenizer, harmless_progress, device
    )
    harmless_means = compute_mean_activations_from_lists(harmless_activations)

    directions = compute_difference_in_means_directions(harmful_means, harmless_means)

    print("Computing per-question direction projections...", flush=True)
    forget_projections = compute_per_question_direction_projections(
        harmful_activations, directions
    )
    retain_projections = compute_per_question_direction_projections(
        harmless_activations, directions
    )

    save_record = {
        "model": model_id,
        "model_key": model_key,
        "harmful_split": HARMFUL_SPLIT,
        "harmless_split": HARMLESS_SPLIT,
        "num_questions_per_split": num_questions,
        "harmful_questions_used": harmful_count,
        "harmless_questions_used": harmless_count,
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "directions": directions,
        "direction_projections": {
            "forget": forget_projections,
            "retain": retain_projections,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_record, output_path)
    print(f"Saved refusal directions to {output_path}", flush=True)

    if projection_plot_layer is not None:
        plot_output_path = projection_plot_output_path
        if plot_output_path is None:
            if model_key is None:
                raise ValueError(
                    "model_key is required to derive the projection plot output path."
                )
            plot_output_path = refusal_direction_projection_plot_path(
                model_key, projection_plot_layer
            )
        print(
            f"Plotting direction projection KDE for layer {projection_plot_layer}...",
            flush=True,
        )
        plot_direction_projection_from_record(
            save_record, projection_plot_layer, plot_output_path
        )

    return save_record
