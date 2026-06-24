"""Output path resolution from config/models.yaml."""

import json
from pathlib import Path

from utils.constants import DIRECTION_SOURCE_CONFABULATION, DIRECTION_SOURCE_CONFIG_KEYS


def resolve_config_output(model_entry, output_key, default_path=None):
    """
    Resolve an output path from model config, with an optional fallback.

    Args:
        model_entry: Model dict from config/models.yaml.
        output_key: Key under model_entry['outputs'].
        default_path: Optional fallback when the key is absent.

    Returns:
        Path string, or None if not configured and no fallback given.
    """
    configured_path = model_entry.get("outputs", {}).get(output_key)
    if configured_path:
        return configured_path
    return default_path


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


def load_probe_answers_by_index(probe_file):
    """
    Load non-ablated model answers from a tofu_probe.py results file.

    Args:
        probe_file: Path to a probe JSON file, or None.

    Returns:
        Tuple of (index-to-answer dict, probe file path used). The dict is empty
        when no probe file is available.
    """
    if probe_file is None:
        return {}, None

    probe_file_path = Path(probe_file)
    if not probe_file_path.is_file():
        print(
            f"Warning: probe file not found at {probe_file_path}; "
            f"probe_answer will be omitted.",
            flush=True,
        )
        return {}, str(probe_file_path)

    probe_record = json.loads(probe_file_path.read_text(encoding="utf-8"))
    probe_answers_by_index = {
        entry["index"]: entry["model_answer"] for entry in probe_record["results"]
    }
    print(
        f"Loaded {len(probe_answers_by_index)} non-ablated answers from {probe_file_path}.",
        flush=True,
    )
    return probe_answers_by_index, str(probe_file_path)


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


def default_sweep_output_path(model_entry, directions_source):
    """
    Derive the default coefficient-sweep output path for a directions source.

    Appends _sweep before the file extension on the normal ablate-and-probe path.

    Args:
        model_entry: Model dict from config/models.yaml.
        directions_source: Either 'refusal' or 'confabulation'.

    Returns:
        Default sweep output path string, or None if the model has no ablate_and_probe output.
    """
    base_output_path = default_ablate_and_probe_output_path(
        model_entry, directions_source
    )
    if base_output_path is None:
        return None

    base_path = Path(base_output_path)
    return str(base_path.with_name(base_path.stem + "_sweep" + base_path.suffix))


def default_probe_direction_output_path(model_entry, model_key):
    """
    Derive the default linear-probe output path for a model.

    Uses outputs.probe_direction from config when present, otherwise falls back to
    results/probe-direction/{model_key}.json.

    Args:
        model_entry: Model dict from config/models.yaml.
        model_key: Config key for the model.

    Returns:
        Default output path string.
    """
    configured_output = model_entry.get("outputs", {}).get("probe_direction")
    if configured_output:
        return configured_output
    return f"results/probe-direction/{model_key}.json"


def default_harvested_answers_path(direction_output_path):
    """
    Derive the default harvested-answers JSON path from a directions output path.

    Args:
        direction_output_path: Path where per-layer directions will be saved.

    Returns:
        Path for the companion harvested-answers JSON file.
    """
    direction_output_path = Path(direction_output_path)
    return direction_output_path.with_name(
        direction_output_path.stem + "_harvested.json"
    )
