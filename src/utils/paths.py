"""Output path resolution from config/models.yaml."""

import json
from pathlib import Path

from utils.constants import (
    ABLATION_METHOD_HOOKS,
    ABLATION_METHOD_ORTHOGONALISATION,
    ABLATION_METHOD_STEER,
    DIRECTION_SOURCE_CONFABULATION,
    DIRECTION_SOURCE_CONFIG_KEYS,
    DIRECTION_SOURCE_REFUSAL,
)

ABLATE_AND_PROBE_RESULTS_DIR = Path("results/ablate-and-probe")


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


def build_ablate_probe_output_path(
    model_key,
    directions_source,
    ablation_method=ABLATION_METHOD_HOOKS,
    is_coefficient_sweep=False,
    steering_layer=None,
    steering_coefficient=None,
):
    """
    Build the default ablate-and-probe output path under results/ablate-and-probe/{model_key}/.

    Filenames omit the model key, start with the intervention type, and end with the
    direction source, e.g. ablate_hooks_refusal.json, ablate_hooks_confab.json,
    negsteer_sweep_layer14_refusal.json, negsteer_layer14_coef2.5_confab.json.

    Args:
        model_key: Short name from config/models.yaml.
        directions_source: Either 'refusal' or 'confabulation'.
        ablation_method: One of hooks, orthogonalisation, or steer.
        is_coefficient_sweep: True when multiple steering coefficients are swept.
        steering_layer: Layer index for steering runs, or None.
        steering_coefficient: Single steering coefficient for non-sweep steer runs.

    Returns:
        Default output path string.
    """
    filename_parts = []

    uses_steering = (
        ablation_method == ABLATION_METHOD_STEER
        or is_coefficient_sweep
        or steering_layer is not None
    )

    if uses_steering:
        filename_parts.append("negsteer")
        if is_coefficient_sweep:
            filename_parts.append("sweep")
            if steering_layer is not None:
                filename_parts.append(f"layer{steering_layer}")
        elif steering_layer is not None:
            filename_parts.append(f"layer{steering_layer}")
            if steering_coefficient is not None:
                coefficient_label = format(steering_coefficient, ".10g")
                filename_parts.append(f"coef{coefficient_label}")
    elif ablation_method == ABLATION_METHOD_ORTHOGONALISATION:
        filename_parts.append("ablate_orthog")
    else:
        filename_parts.append("ablate_hooks")

    if directions_source == DIRECTION_SOURCE_CONFABULATION:
        filename_parts.append("confab")
    else:
        filename_parts.append("refusal")

    filename = "_".join(filename_parts) + ".json"
    return str(ABLATE_AND_PROBE_RESULTS_DIR / model_key / filename)


def default_ablate_and_probe_output_path(
    model_key, directions_source, ablation_method=ABLATION_METHOD_HOOKS
):
    """
    Derive the default ablate-and-probe output path for hooks or orthogonalisation runs.

    Args:
        model_key: Short name from config/models.yaml.
        directions_source: Either 'refusal' or 'confabulation'.
        ablation_method: One of hooks or orthogonalisation.

    Returns:
        Default output path string.
    """
    return build_ablate_probe_output_path(
        model_key, directions_source, ablation_method=ablation_method
    )


def default_sweep_output_path(
    model_key, directions_source, steering_layer=None
):
    """
    Derive the default multi-coefficient sweep output path for a model config key.

    Args:
        model_key: Short name from config/models.yaml.
        directions_source: Either 'refusal' or 'confabulation'.
        steering_layer: Layer index for the steering sweep.

    Returns:
        Default sweep output path string.
    """
    return build_ablate_probe_output_path(
        model_key,
        directions_source,
        ablation_method=ABLATION_METHOD_STEER,
        is_coefficient_sweep=True,
        steering_layer=steering_layer,
    )


def append_layer_suffix(output_path, steering_layer):
    """
    Append _layer{N} before the file extension for steering runs.

    Args:
        output_path: Base output path string, or None.
        steering_layer: Layer index used for steering, or None to leave unchanged.

    Returns:
        Path string with layer suffix, or None if output_path was None.
    """
    if output_path is None or steering_layer is None:
        return output_path

    output_file_path = Path(output_path)
    return str(
        output_file_path.with_name(
            output_file_path.stem + f"_layer{steering_layer}" + output_file_path.suffix
        )
    )


def append_coefficient_suffix(output_path, steering_coefficient):
    """
    Append _coef{value} before the file extension for single-coefficient steering runs.

    Args:
        output_path: Base output path string, or None.
        steering_coefficient: Steering coefficient value, or None to leave unchanged.

    Returns:
        Path string with coefficient suffix, or None if output_path was None.
    """
    if output_path is None or steering_coefficient is None:
        return output_path

    coefficient_label = format(steering_coefficient, ".10g")
    output_file_path = Path(output_path)
    return str(
        output_file_path.with_name(
            output_file_path.stem
            + f"_coef{coefficient_label}"
            + output_file_path.suffix
        )
    )


def default_probe_refusal_direction_output_path(model_entry, model_key):
    """
    Derive the default forget-vs-retain linear-probe output path for a model.

    Uses outputs.probe_refusal_direction from config when present, otherwise falls back
    to results/probe-refusal-direction/{model_key}.json.

    Args:
        model_entry: Model dict from config/models.yaml.
        model_key: Config key for the model.

    Returns:
        Default output path string.
    """
    configured_output = model_entry.get("outputs", {}).get("probe_refusal_direction")
    if configured_output:
        return configured_output
    return f"results/probe-refusal-direction/{model_key}.json"


def default_probe_confab_direction_output_path(model_entry, model_key):
    """
    Derive the default confab-vs-correct linear-probe output path for a model.

    Uses outputs.probe_confab_direction from config when present, otherwise falls back
    to results/probe-confab-direction/{model_key}.json.

    Args:
        model_entry: Model dict from config/models.yaml.
        model_key: Config key for the model.

    Returns:
        Default output path string.
    """
    configured_output = model_entry.get("outputs", {}).get("probe_confab_direction")
    if configured_output:
        return configured_output
    return f"results/probe-confab-direction/{model_key}.json"


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
