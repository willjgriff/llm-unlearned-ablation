"""Load model keys from config/models.yaml."""
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "models.yaml"


def get_model(model_key):
    """
    Look up a model entry by its config key.

    Args:
        model_key: Short name defined in config/models.yaml.

    Returns:
        Dict with hf_id, outputs, and optional directions_file.
    """
    models = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["models"]
    if model_key not in models:
        available = ", ".join(sorted(models))
        raise KeyError(f"Unknown model key '{model_key}'. Available: {available}")
    return models[model_key]
