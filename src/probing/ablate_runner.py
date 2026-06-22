"""Orchestrate directional intervention and forget-set probing."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from ablation import (
    apply_weight_orthogonalisation,
    register_ablation_hooks_on_all_layers,
    register_steering_hook,
    remove_ablation_hooks,
    restore_original_weights,
)
from utils.constants import (
    ABLATION_METHOD_HOOKS,
    ABLATION_METHOD_ORTHOGONALISATION,
    ABLATION_METHOD_STEER,
    FORGET_SPLIT,
)
from utils.directions import load_all_direction_vectors, print_direction_norms
from utils.inference import generate_answer, load_model_and_tokenizer


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


def apply_ablation_method(
    model,
    direction_vectors,
    ablation_method,
    directions_source,
    device,
    model_dtype,
    steering_layer=None,
    steering_coefficient=1.0,
):
    """
    Apply the chosen directional intervention and return state for cleanup.

    Args:
        model: Loaded causal LM in eval mode.
        direction_vectors: List of per-layer direction tensors.
        ablation_method: One of 'hooks', 'orthogonalisation', or 'steer'.
        directions_source: Direction type label for logging ('refusal' or 'confabulation').
        device: Torch device string.
        model_dtype: Model weight dtype.
        steering_layer: Layer index for steering (required when method is steer).
        steering_coefficient: Scalar multiplier for steering (steer only).

    Returns:
        Tuple of (ablation_handles, saved_weights). One is set depending on method.
    """
    num_layers = len(model.model.layers)
    ablation_handles = None
    saved_weights = None

    if ablation_method == ABLATION_METHOD_HOOKS:
        ablation_handles = register_ablation_hooks_on_all_layers(
            model, direction_vectors, device, model_dtype
        )
        print(
            f"Ablating {directions_source} direction at all {num_layers} layers "
            f"via {ablation_method}.",
            flush=True,
        )
    elif ablation_method == ABLATION_METHOD_ORTHOGONALISATION:
        saved_weights = apply_weight_orthogonalisation(
            model, direction_vectors, device, model_dtype
        )
        print(
            f"Ablating {directions_source} direction at all {num_layers} layers "
            f"via {ablation_method}.",
            flush=True,
        )
    elif ablation_method == ABLATION_METHOD_STEER:
        if steering_layer is None:
            raise ValueError(
                "steering_layer is required when ablation_method is 'steer'."
            )
        ablation_handles = register_steering_hook(
            model,
            direction_vectors,
            steering_layer,
            steering_coefficient,
            device,
            model_dtype,
        )
        print(
            f"Steering {directions_source} direction at layer {steering_layer} "
            f"(coefficient {steering_coefficient}) via {ablation_method}.",
            flush=True,
        )
    else:
        raise ValueError(
            f"Unknown ablation method '{ablation_method}'; expected "
            f"'{ABLATION_METHOD_HOOKS}', '{ABLATION_METHOD_ORTHOGONALISATION}', "
            f"or '{ABLATION_METHOD_STEER}'."
        )

    return ablation_handles, saved_weights


def cleanup_ablation(model, ablation_handles, saved_weights):
    """
    Undo hooks or restored orthogonalised weights after probing.

    Args:
        model: Loaded causal LM that was intervened on.
        ablation_handles: Hook handles from apply_ablation_method, if any.
        saved_weights: Saved weight copies from orthogonalisation, if any.
    """
    if ablation_handles is not None:
        remove_ablation_hooks(ablation_handles)
    if saved_weights is not None:
        restore_original_weights(model, saved_weights)


def ablate_and_probe(
    model_id,
    directions_file,
    num_questions,
    max_new_tokens,
    ablation_method=ABLATION_METHOD_HOOKS,
    directions_source="refusal",
    steering_layer=None,
    steering_coefficient=1.0,
    probe_file=None,
    output_path=None,
    model_key=None,
):
    """
    Apply directional intervention and probe forget-set questions.

    Args:
        model_id: Hugging Face model id or local path.
        directions_file: Path to saved per-layer direction vectors.
        num_questions: Number of questions to probe from the start of the forget10 split.
        max_new_tokens: Maximum tokens to generate per question.
        ablation_method: Either 'hooks', 'orthogonalisation', or 'steer'.
        directions_source: Which direction type was loaded ('refusal' or 'confabulation').
        steering_layer: Layer index for single-layer steering (required when method is steer).
        steering_coefficient: Scalar multiplier for the steering direction (steer only).
        probe_file: Optional path to a tofu_probe.py JSON file for non-ablated answers.
        output_path: Optional path to write structured JSON results.
        model_key: Optional config key from config/models.yaml.

    Returns:
        Dict containing run metadata and per-question results.
    """
    model, tokenizer, device, model_dtype = load_model_and_tokenizer(model_id)

    direction_vectors = load_all_direction_vectors(directions_file)
    num_layers_ablated = len(model.model.layers)
    if len(direction_vectors) != num_layers_ablated:
        raise ValueError(
            f"Directions file has {len(direction_vectors)} layers but model has "
            f"{num_layers_ablated}."
        )

    ablation_handles, saved_weights = apply_ablation_method(
        model,
        direction_vectors,
        ablation_method,
        directions_source,
        device,
        model_dtype,
        steering_layer=steering_layer,
        steering_coefficient=steering_coefficient,
    )

    sampled_layer_indices = list(range(0, num_layers_ablated, 4))
    print_direction_norms(direction_vectors, sampled_layer_indices)

    probe_answers_by_index, probe_file_used = load_probe_answers_by_index(probe_file)

    dataset = load_dataset("locuslab/TOFU", FORGET_SPLIT)["train"]
    question_count = min(num_questions, len(dataset))
    print(
        f"Probing {question_count} questions from TOFU '{FORGET_SPLIT}' "
        f"({len(dataset)} available)...",
        flush=True,
    )

    results = []
    progress = tqdm(
        range(question_count),
        desc="Probing",
        unit="question",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    try:
        for index in progress:
            question = dataset[index]["question"]
            ground_truth_answer = dataset[index]["answer"]

            progress.set_postfix_str(f"asking {index + 1}/{question_count}", refresh=True)
            model_answer = generate_answer(
                model, tokenizer, question, device, max_new_tokens
            )
            progress.set_postfix_str(f"answered {index + 1}/{question_count}", refresh=True)

            result_entry = {
                "index": index,
                "question": question,
                "ground_truth": ground_truth_answer,
                "model_answer": model_answer,
            }
            if index in probe_answers_by_index:
                result_entry["probe_answer"] = probe_answers_by_index[index]
            results.append(result_entry)
    finally:
        cleanup_ablation(model, ablation_handles, saved_weights)

    run_record = {
        "model": model_id,
        "model_key": model_key,
        "directions_file": directions_file,
        "directions_source": directions_source,
        "ablation_method": ablation_method,
        "steering_layer": steering_layer,
        "steering_coefficient": steering_coefficient,
        "probe_file": probe_file_used,
        "num_layers_ablated": num_layers_ablated,
        "split": FORGET_SPLIT,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": question_count,
        "results": results,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(run_record, indent=2) + "\n")
        print(f"Saved results to {output_path}", flush=True)

    return run_record
