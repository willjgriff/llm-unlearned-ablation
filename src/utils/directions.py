"""Load and inspect saved per-layer direction vector files."""
from pathlib import Path

import torch


def load_all_direction_vectors(directions_file):
    """
    Load all per-layer direction vectors from a saved directions file.

    Args:
        directions_file: Path to a .pt file from direction-calculation scripts.

    Returns:
        List of direction tensors, one per layer.
    """
    saved_record = torch.load(directions_file, map_location="cpu", weights_only=False)
    return saved_record["directions"]


def print_direction_norms(direction_vectors, layer_indices):
    """
    Print the L2 norm of direction vectors at selected layer indices.

    Args:
        direction_vectors: List of direction tensors, one per layer.
        layer_indices: Layer indices to sample and print norms for.
    """
    for layer_index in layer_indices:
        direction_norm = direction_vectors[layer_index].norm().item()
        print(f"Layer {layer_index} direction norm: {direction_norm:.4f}", flush=True)
