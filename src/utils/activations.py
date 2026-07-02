"""Residual stream activation extraction helpers."""

import numpy as np
import torch


def tokenize_question(tokenizer, question, device):
    """
    Wrap a TOFU question in the chat template and move token ids to the device.

    Args:
        tokenizer: Loaded tokenizer with a chat template.
        question: User question text from the TOFU dataset.
        device: Torch device string (cuda, mps, or cpu).

    Returns:
        Input token ids tensor of shape (1, sequence_length).
    """
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)


def build_answer_sequence_token_ids(tokenizer, question, answer_text):
    """
    Build the full prefix-plus-answer token sequence used during generation.

    Uses the chat-template generation prompt as prefix, then appends tokenised answer
    text without special tokens so activations match on-distribution generation.

    Args:
        tokenizer: Loaded tokenizer with chat template.
        question: User question text from the TOFU dataset.
        answer_text: Answer text to append after the generation prompt.

    Returns:
        Token ids tensor of shape (1, sequence_length).
    """
    messages = [{"role": "user", "content": question}]
    prefix_token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if not isinstance(prefix_token_ids, torch.Tensor):
        prefix_token_ids = prefix_token_ids["input_ids"]

    answer_encoding = tokenizer(
        answer_text, add_special_tokens=False, return_tensors="pt"
    )
    answer_token_ids = answer_encoding["input_ids"]
    return torch.cat([prefix_token_ids, answer_token_ids], dim=-1)


def extract_last_token_activations(outputs, num_layers):
    """
    Extract last-token residual activations at every layer from model outputs.

    Args:
        outputs: Model forward pass result with hidden_states populated.
        num_layers: Number of transformer layers in the model.

    Returns:
        List of last-token activation tensors, one per layer.
    """
    return [
        outputs.hidden_states[layer_index + 1][0, -1, :].float().cpu()
        for layer_index in range(num_layers)
    ]


def collect_last_token_activation_sums(model, tokenizer, questions, device):
    """
    Accumulate last-token residual-stream activations at every layer for questions.

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Matching tokenizer with chat template.
        questions: Iterable of question strings.
        device: Torch device string (cuda, mps, or cpu).

    Returns:
        Tuple of (activation sums per layer, number of questions processed).
    """
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    activation_sums = [torch.zeros(hidden_size) for _ in range(num_layers)]
    question_count = 0

    for question in questions:
        input_token_ids = tokenize_question(tokenizer, question, device)

        with torch.no_grad():
            outputs = model(input_token_ids, output_hidden_states=True)

        for layer_index, last_token_activation in enumerate(
            extract_last_token_activations(outputs, num_layers)
        ):
            activation_sums[layer_index] += last_token_activation
        question_count += 1

    return activation_sums, question_count


def collect_last_token_activations(model, tokenizer, questions, device):
    """
    Collect last-token residual-stream activations at every layer for each question.

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Matching tokenizer with chat template.
        questions: Iterable of question strings.
        device: Torch device string (cuda, mps, or cpu).

    Returns:
        Tuple of (per-layer activation lists, number of questions processed). Each
        layer list contains one hidden-state vector per question.
    """
    num_layers = model.config.num_hidden_layers
    per_layer_activations = [[] for _ in range(num_layers)]
    question_count = 0

    for question in questions:
        input_token_ids = tokenize_question(tokenizer, question, device)

        with torch.no_grad():
            outputs = model(input_token_ids, output_hidden_states=True)

        for layer_index, last_token_activation in enumerate(
            extract_last_token_activations(outputs, num_layers)
        ):
            per_layer_activations[layer_index].append(last_token_activation.numpy())
        question_count += 1

    return per_layer_activations, question_count


def collect_last_token_activation_sums_for_sequences(
    model, tokenizer, entries, device, answer_field
):
    """
    Accumulate last-token activations for full prefix-plus-answer sequences.

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Matching tokenizer with chat template.
        entries: Sequence entries with question and answer text fields.
        device: Torch device string (cuda, mps, or cpu).
        answer_field: Entry key for answer text ('model_answer' or 'ground_truth').

    Returns:
        Tuple of (activation sums per layer, number of sequences processed).
    """
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    activation_sums = [torch.zeros(hidden_size) for _ in range(num_layers)]
    sequence_count = 0

    for entry in entries:
        full_sequence_token_ids = build_answer_sequence_token_ids(
            tokenizer, entry["question"], entry[answer_field]
        ).to(device)

        with torch.no_grad():
            outputs = model(full_sequence_token_ids, output_hidden_states=True)

        for layer_index, last_token_activation in enumerate(
            extract_last_token_activations(outputs, num_layers)
        ):
            activation_sums[layer_index] += last_token_activation
        sequence_count += 1

    return activation_sums, sequence_count


def collect_last_token_activations_for_sequences(
    model, tokenizer, entries, device, answer_field
):
    """
    Collect last-token activations for full prefix-plus-answer sequences.

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Matching tokenizer with chat template.
        entries: Sequence entries with question and answer text fields.
        device: Torch device string (cuda, mps, or cpu).
        answer_field: Entry key for answer text ('model_answer' or 'ground_truth').

    Returns:
        Tuple of (per-layer activation lists, number of sequences processed). Each
        layer list contains one hidden-state vector per sequence.
    """
    num_layers = model.config.num_hidden_layers
    per_layer_activations = [[] for _ in range(num_layers)]
    sequence_count = 0

    for entry in entries:
        full_sequence_token_ids = build_answer_sequence_token_ids(
            tokenizer, entry["question"], entry[answer_field]
        ).to(device)

        with torch.no_grad():
            outputs = model(full_sequence_token_ids, output_hidden_states=True)

        for layer_index, last_token_activation in enumerate(
            extract_last_token_activations(outputs, num_layers)
        ):
            per_layer_activations[layer_index].append(last_token_activation.numpy())
        sequence_count += 1

    return per_layer_activations, sequence_count


def compute_mean_activations(activation_sums, sample_count):
    """
    Convert per-layer activation sums into per-layer mean vectors.

    Args:
        activation_sums: List of summed activation tensors, one per layer.
        sample_count: Number of samples included in the sums.

    Returns:
        List of mean activation tensors, one per layer.
    """
    return [
        activation_sums[layer_index] / sample_count
        for layer_index in range(len(activation_sums))
    ]


def compute_mean_activations_from_lists(per_layer_activations):
    """
    Convert per-question activation lists into per-layer mean vectors.

    Args:
        per_layer_activations: List of layers, each containing one activation
            vector per question.

    Returns:
        List of mean activation tensors, one per layer.
    """
    return [
        torch.tensor(
            np.stack(per_layer_activations[layer_index]).mean(axis=0),
            dtype=torch.float32,
        )
        for layer_index in range(len(per_layer_activations))
    ]


def compute_difference_in_means_directions(positive_means, negative_means):
    """
    Subtract negative mean activations from positive mean activations per layer.

    Args:
        positive_means: Mean last-token activations for the positive class.
        negative_means: Mean last-token activations for the negative class.

    Returns:
        List of raw difference-in-means direction vectors, one per layer.
    """
    return [
        positive_means[layer_index] - negative_means[layer_index]
        for layer_index in range(len(positive_means))
    ]


def compute_per_question_direction_projections(
    per_layer_activations, direction_vectors
):
    """
    Compute per-question dot products with a direction vector at every layer.

    Args:
        per_layer_activations: List of layers, each containing one activation
            vector per question.
        direction_vectors: List of direction tensors, one per layer.

    Returns:
        List of per-layer lists of scalar dot-product values, one per question.
    """
    per_layer_projections = []
    for layer_index, layer_activations in enumerate(per_layer_activations):
        direction_vector = direction_vectors[layer_index].float()
        layer_projections = []
        for activation in layer_activations:
            activation_tensor = torch.tensor(activation, dtype=torch.float32)
            layer_projections.append(
                torch.dot(activation_tensor, direction_vector).item()
            )
        per_layer_projections.append(layer_projections)
    return per_layer_projections
