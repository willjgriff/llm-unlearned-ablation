"""Resolve output and input paths from model config and CLI overrides."""
from pathlib import Path

from utils.constants import (
    DIRECTION_SOURCE_CONFABULATION,
    DIRECTION_SOURCE_CONFIG_KEYS,
)


def resolve_directions_file(model_entry, directions_file_argument, directions_source):
    """
    Resolve the directions file path from CLI args and model config.

    Args:
        model_entry: Model dict from config/models.yaml.
        directions_file_argument: Explicit --directions-file path, if any.
        directions_source: Either 'refusal' or 'confabulation'.

    Returns:
        Path string to the directions .pt file.
    """
    if directions_file_argument:
        return directions_file_argument
    if model_entry.get("directions_file"):
        return model_entry["directions_file"]

    config_output_key = DIRECTION_SOURCE_CONFIG_KEYS[directions_source]
    model_outputs = model_entry["outputs"]
    if config_output_key not in model_outputs:
        raise KeyError(
            f"Model '{model_entry.get('hf_id', 'unknown')}' has no "
            f"outputs.{config_output_key} in config/models.yaml."
        )
    return model_outputs[config_output_key]


def resolve_probe_file(model_entry, probe_file_argument):
    """
    Resolve the probe results JSON path from CLI args and model config.

    Args:
        model_entry: Model dict from config/models.yaml.
        probe_file_argument: Explicit --probe-file path, if any.

    Returns:
        Path string to the probe JSON file, or None if not configured.
    """
    if probe_file_argument:
        return probe_file_argument
    return model_entry.get("outputs", {}).get("probe")


def default_ablate_and_probe_output_path(model_entry, directions_source):
    """
    Derive the default ablate-and-probe output path for a directions source.

    Confabulation runs use a separate _confab suffix so they do not overwrite
    refusal-direction ablation results.

    Args:
        model_entry: Model dict from config/models.yaml.
        directions_source: Either 'refusal' or 'confabulation'.

    Returns:
        Default output path string, or None if the model has no ablate_and_probe output.
    """
    if "ablate_and_probe" not in model_entry["outputs"]:
        return None

    base_output_path = Path(model_entry["outputs"]["ablate_and_probe"])
    if directions_source == DIRECTION_SOURCE_CONFABULATION:
        return str(
            base_output_path.with_name(
                base_output_path.stem + "_confab" + base_output_path.suffix
            )
        )
    return str(base_output_path)
