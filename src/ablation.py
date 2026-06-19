"""
ablation.py - directional refusal ablation via forward hooks or weight orthogonalisation.
"""
import torch


def make_directional_ablation_hook(direction_vector, device, model_dtype):
    """
    Build a forward hook that removes the component along a unit direction vector.

    Applies x' = x - (r_hat r_hat^T x) at every token position in the layer output.

    Args:
        direction_vector: Raw direction tensor of shape (hidden_size,).
        device: Torch device for the normalised direction.
        model_dtype: Model dtype to cast the direction to.

    Returns:
        Forward hook callable for register_forward_hook.
    """
    direction_hat = direction_vector / direction_vector.norm()
    direction_hat = direction_hat.to(device=device, dtype=model_dtype)

    def ablation_hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        projection_coefficients = torch.matmul(hidden_states, direction_hat)
        projection = projection_coefficients.unsqueeze(-1) * direction_hat
        modified_hidden_states = hidden_states - projection
        if isinstance(output, tuple):
            return (modified_hidden_states,) + output[1:]
        return modified_hidden_states

    return ablation_hook


def register_ablation_hooks_on_all_layers(model, direction_vectors, device, model_dtype):
    """
    Register a directional ablation hook on every transformer layer.

    The Arditi et al. paper ablates the direction at every layer simultaneously.
    Hooking only one layer allows the model to re-encode the deflection signal
    at all other layers, which is why single-layer ablation does not work.

    Args:
        model: Loaded causal LM in eval mode.
        direction_vectors: List of direction tensors, one per layer (from load_all_direction_vectors).
        device: Torch device string.
        model_dtype: Model dtype to cast direction vectors to.

    Returns:
        List of hook handles — pass to remove_ablation_hooks when done.
    """
    num_layers = len(model.model.layers)
    if len(direction_vectors) != num_layers:
        raise ValueError(
            f"Directions file has {len(direction_vectors)} layers but model has {num_layers}."
        )

    ablation_handles = []
    for layer_index in range(num_layers):
        handle = model.model.layers[layer_index].register_forward_hook(
            make_directional_ablation_hook(
                direction_vectors[layer_index], device, model_dtype
            )
        )
        ablation_handles.append(handle)

    return ablation_handles


def remove_ablation_hooks(ablation_handles):
    """
    Remove all registered ablation hooks.

    Args:
        ablation_handles: List of hook handles returned by register_ablation_hooks_on_all_layers.
    """
    for handle in ablation_handles:
        handle.remove()


def _orthogonalise_residual_output_weight(weight_matrix, direction_hat):
    """
    Orthogonalise an output weight matrix: W' = W - r_hat r_hat^T W.

    For matrices of shape (out_features, in_features) where the direction lives in
    out_features space (e.g. o_proj, down_proj).

    Args:
        weight_matrix: Weight tensor of shape (out_features, in_features).
        direction_hat: Unit-normalised direction vector of shape (out_features,).

    Returns:
        Orthogonalised weight tensor in float32.
    """
    weight_float32 = weight_matrix.to(torch.float32)
    direction_hat_float32 = direction_hat.to(torch.float32)
    direction_projection = torch.outer(
        direction_hat_float32, torch.matmul(direction_hat_float32, weight_float32)
    )
    return weight_float32 - direction_projection


def _orthogonalise_embedding_weight(embedding_weight, direction_hat):
    """
    Orthogonalise embedding rows against a direction in hidden space.

    embed_tokens.weight has shape (vocab_size, hidden_size); each row is a hidden
    vector written to the residual stream, so the direction is applied along rows.

    Args:
        embedding_weight: Embedding weight tensor of shape (vocab_size, hidden_size).
        direction_hat: Unit-normalised direction vector of shape (hidden_size,).

    Returns:
        Orthogonalised embedding weight tensor in float32.
    """
    weight_float32 = embedding_weight.to(torch.float32)
    direction_hat_float32 = direction_hat.to(torch.float32)
    row_projection_coefficients = torch.matmul(weight_float32, direction_hat_float32)
    direction_projection = torch.outer(row_projection_coefficients, direction_hat_float32)
    return weight_float32 - direction_projection


def _select_strongest_direction_layer(direction_vectors):
    """
    Pick the layer index whose direction vector has the largest L2 norm.

    Args:
        direction_vectors: List of direction tensors, one per layer.

    Returns:
        Tuple of (selected_layer_index, raw_direction_vector, direction_norm).
    """
    selected_layer_index = max(
        range(len(direction_vectors)),
        key=lambda layer_index: direction_vectors[layer_index].norm().item(),
    )
    selected_direction = direction_vectors[selected_layer_index]
    selected_direction_norm = selected_direction.norm().item()
    return selected_layer_index, selected_direction, selected_direction_norm


def _collect_residual_stream_weights(model):
    """
    Collect every weight parameter that writes to the residual stream.

    Args:
        model: Loaded causal LM in eval mode.

    Returns:
        List of (weight_parameter, orthogonalise_fn) tuples.
    """
    residual_stream_weights = [
        (model.model.embed_tokens.weight, _orthogonalise_embedding_weight),
    ]
    for transformer_layer in model.model.layers:
        residual_stream_weights.append(
            (transformer_layer.self_attn.o_proj.weight, _orthogonalise_residual_output_weight)
        )
        residual_stream_weights.append(
            (transformer_layer.mlp.down_proj.weight, _orthogonalise_residual_output_weight)
        )
    return residual_stream_weights


def apply_weight_orthogonalisation(model, direction_vectors, device, model_dtype):
    """
    Orthogonalise all residual-stream output weights against one shared direction.

    Selects the per-layer direction with the largest L2 norm, unit-normalises it,
    then applies W' = W - r_hat r_hat^T W to embed_tokens, every layer's o_proj,
    and every layer's down_proj. Matches the Arditi et al. weight-orthogonalisation
    method, which is provably equivalent to all-layer forward-hook ablation.

    Args:
        model: Loaded causal LM in eval mode.
        direction_vectors: List of direction tensors, one per layer.
        device: Torch device string where model weights live.
        model_dtype: Model dtype to cast orthogonalised weights back to.

    Returns:
        List of dicts with keys 'parameter' and 'saved_value' for restore_original_weights.
    """
    num_layers = len(model.model.layers)
    if len(direction_vectors) != num_layers:
        raise ValueError(
            f"Directions file has {len(direction_vectors)} layers but model has {num_layers}."
        )

    selected_layer_index, selected_direction, selected_direction_norm = (
        _select_strongest_direction_layer(direction_vectors)
    )
    print(
        f"Weight orthogonalisation: selected layer {selected_layer_index} "
        f"(direction norm {selected_direction_norm:.4f}).",
        flush=True,
    )

    direction_hat = selected_direction / selected_direction.norm()
    direction_hat = direction_hat.to(device=device)

    saved_weights = []
    with torch.no_grad():
        for weight_parameter, orthogonalise_function in _collect_residual_stream_weights(
            model
        ):
            saved_weights.append(
                {
                    "parameter": weight_parameter,
                    "saved_value": weight_parameter.data.clone().cpu(),
                }
            )
            orthogonalised_weight = orthogonalise_function(
                weight_parameter.data, direction_hat
            )
            weight_parameter.data.copy_(
                orthogonalised_weight.to(device=weight_parameter.device, dtype=model_dtype)
            )

    return saved_weights


def restore_original_weights(model, saved_weights):
    """
    Restore model weights saved before apply_weight_orthogonalisation.

    Args:
        model: Loaded causal LM whose weights were orthogonalised.
        saved_weights: List returned by apply_weight_orthogonalisation.
    """
    with torch.no_grad():
        for saved_weight_entry in saved_weights:
            weight_parameter = saved_weight_entry["parameter"]
            original_weight = saved_weight_entry["saved_value"].to(
                device=weight_parameter.device, dtype=weight_parameter.dtype
            )
            weight_parameter.data.copy_(original_weight)
