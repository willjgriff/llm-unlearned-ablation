"""Ablation probing and coefficient sweep runners."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    DIRECTION_SOURCE_REFUSAL,
    FORGET_SPLIT,
)
from utils.directions_io import (
    load_all_direction_vectors,
    print_direction_norms,
)
from utils.inference import generate_answer
from utils.metrics import (
    attach_ablate_summaries_before_results,
    compute_rouge_l,
    summarize_flat_results,
    summarize_sweep_results,
)
from utils.model_loading import load_model_and_tokenizer
from utils.paths import load_probe_answers_by_index, load_probe_summary
from utils.tofu_data import load_forget_split_dataset


def group_sweep_results_by_question(coefficient_runs):
    """
    Group per-coefficient sweep results by question index.

    Each output entry lists the question once and collects all model answers
    from every coefficient run below it. When only one coefficient was run,
    returns the flat per-question results unchanged.

    Args:
        coefficient_runs: List of dicts with steering_coefficient and results keys.

    Returns:
        List of per-question dicts. Multiple coefficients use model_answers; a
        single coefficient uses model_answer alongside ground_truth and probe_answer.
    """
    if not coefficient_runs:
        return []

    if len(coefficient_runs) == 1:
        return coefficient_runs[0]["results"]

    question_count = len(coefficient_runs[0]["results"])
    grouped_results = []
    for question_index in range(question_count):
        base_entry = coefficient_runs[0]["results"][question_index]
        grouped_entry = {
            "index": base_entry["index"],
            "question": base_entry["question"],
            "ground_truth": base_entry["ground_truth"],
        }
        if "probe_answer" in base_entry:
            grouped_entry["probe_answer"] = base_entry["probe_answer"]

        model_answers = []
        for coefficient_run in coefficient_runs:
            coefficient_result = coefficient_run["results"][question_index]
            model_answers.append(
                {
                    "steering_coefficient": coefficient_run["steering_coefficient"],
                    "model_answer": coefficient_result["model_answer"],
                    "rouge_l": coefficient_result["rouge_l"],
                }
            )
        grouped_entry["model_answers"] = model_answers
        grouped_entry["max_rouge_l"] = max(
            answer["rouge_l"] for answer in model_answers
        )
        grouped_results.append(grouped_entry)

    return grouped_results


def run_coefficient_sweep(
    model_id,
    directions_file,
    num_questions,
    max_new_tokens,
    directions_source,
    steering_layer,
    steering_coefficients,
    repetition_penalty=1.0,
    probe_file=None,
    output_path=None,
    model_key=None,
):
    """
    Run steering probes across multiple coefficients in one pass over the model.

    Loads the model and dataset once, then for each coefficient registers a steering
    hook, probes all questions, removes hooks, and collects results into a single
    JSON file with one entry per question and all coefficient answers grouped below.

    Args:
        model_id: Hugging Face model id or local path.
        directions_file: Path to saved per-layer direction vectors.
        num_questions: Number of questions to probe from the start of the forget10 split.
        max_new_tokens: Maximum tokens to generate per question.
        directions_source: Which direction type was loaded ('refusal' or 'confabulation').
        steering_layer: Layer index for single-layer steering.
        steering_coefficients: List of steering coefficient values to sweep.
        repetition_penalty: Penalty for repeating tokens during generation (1.0 = no penalty).
        probe_file: Optional path to a tofu_probe.py JSON file for non-ablated answers.
        output_path: Optional path to write structured JSON results.
        model_key: Optional config key from config/models.yaml.

    Returns:
        Dict containing run metadata and per-question results grouped by coefficient.
    """
    model, tokenizer, device, model_dtype = load_model_and_tokenizer(model_id)

    direction_vectors = load_all_direction_vectors(directions_file)
    num_layers_ablated = len(model.model.layers)
    if len(direction_vectors) != num_layers_ablated:
        raise ValueError(
            f"Directions file has {len(direction_vectors)} layers but model has "
            f"{num_layers_ablated}."
        )

    sampled_layer_indices = list(range(0, num_layers_ablated, 4))
    print_direction_norms(direction_vectors, sampled_layer_indices)

    probe_answers_by_index, probe_file_used = load_probe_answers_by_index(probe_file)

    dataset, question_count = load_forget_split_dataset(num_questions, FORGET_SPLIT)
    print(
        f"Probing {question_count} questions from TOFU '{FORGET_SPLIT}' "
        f"({len(dataset)} available) across {len(steering_coefficients)} coefficients...",
        flush=True,
    )

    coefficient_runs = []
    for steering_coefficient in steering_coefficients:
        print(
            f"Steering {directions_source} direction at layer {steering_layer} "
            f"(coefficient {steering_coefficient}) via {ABLATION_METHOD_STEER}.",
            flush=True,
        )
        ablation_handles = register_steering_hook(
            model,
            direction_vectors,
            steering_layer,
            steering_coefficient,
            device,
            model_dtype,
        )

        coefficient_results = []
        progress = tqdm(
            range(question_count),
            desc=f"Probing (coef={steering_coefficient})",
            unit="question",
            file=sys.stderr,
            dynamic_ncols=True,
        )
        try:
            for index in progress:
                question = dataset[index]["question"]
                ground_truth_answer = dataset[index]["answer"]

                progress.set_postfix_str(
                    f"asking {index + 1}/{question_count}", refresh=True
                )
                model_answer = generate_answer(
                    model,
                    tokenizer,
                    question,
                    device,
                    max_new_tokens,
                    repetition_penalty=repetition_penalty,
                )
                progress.set_postfix_str(
                    f"answered {index + 1}/{question_count}", refresh=True
                )

                result_entry = {
                    "index": index,
                    "question": question,
                    "ground_truth": ground_truth_answer,
                    "model_answer": model_answer,
                    "rouge_l": compute_rouge_l(
                        ground_truth_answer, model_answer
                    ),
                }
                if index in probe_answers_by_index:
                    result_entry["probe_answer"] = probe_answers_by_index[index]
                coefficient_results.append(result_entry)
        finally:
            remove_ablation_hooks(ablation_handles)

        coefficient_runs.append(
            {
                "steering_coefficient": steering_coefficient,
                "results": coefficient_results,
            }
        )

    grouped_results = group_sweep_results_by_question(coefficient_runs)
    is_multi_coefficient_sweep = len(coefficient_runs) > 1
    if is_multi_coefficient_sweep:
        summary = summarize_sweep_results(coefficient_runs, grouped_results)
    else:
        summary = summarize_flat_results(grouped_results)

    sweep_record = {
        "model": model_id,
        "model_key": model_key,
        "directions_file": directions_file,
        "directions_source": directions_source,
        "ablation_method": ABLATION_METHOD_STEER,
        "steering_layer": steering_layer,
        "repetition_penalty": repetition_penalty,
        "probe_file": probe_file_used,
        "num_layers_ablated": num_layers_ablated,
        "split": FORGET_SPLIT,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": question_count,
        "results": grouped_results,
    }
    attach_ablate_summaries_before_results(
        sweep_record,
        summary,
        load_probe_summary(probe_file_used, question_count),
    )
    if len(steering_coefficients) == 1:
        sweep_record["steering_coefficient"] = steering_coefficients[0]
    else:
        sweep_record["steering_coefficients"] = steering_coefficients

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(sweep_record, indent=2) + "\n")
        print(f"Saved results to {output_path}", flush=True)

    return sweep_record


def ablate_and_probe(
    model_id,
    directions_file,
    num_questions,
    max_new_tokens,
    ablation_method=ABLATION_METHOD_HOOKS,
    directions_source=DIRECTION_SOURCE_REFUSAL,
    steering_layer=None,
    steering_coefficient=1.0,
    repetition_penalty=1.0,
    probe_file=None,
    output_path=None,
    model_key=None,
):
    """
    Ablate the refusal direction at every layer and probe forget-set questions.

    Args:
        model_id: Hugging Face model id or local path.
        directions_file: Path to saved per-layer direction vectors.
        num_questions: Number of questions to probe from the start of the forget10 split.
        max_new_tokens: Maximum tokens to generate per question.
        ablation_method: Either 'hooks' (forward-hook ablation), 'orthogonalisation'
            (in-place weight orthogonalisation), or 'steer' (single-layer negative
            steering via activation subtraction).
        directions_source: Which direction type was loaded ('refusal' or 'confabulation').
        steering_layer: Layer index for single-layer steering (required when method is steer).
        steering_coefficient: Scalar multiplier for the steering direction (steer only).
        repetition_penalty: Penalty for repeating tokens during generation (1.0 = no penalty).
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

    ablation_handles = None
    saved_weights = None
    if ablation_method == ABLATION_METHOD_HOOKS:
        ablation_handles = register_ablation_hooks_on_all_layers(
            model, direction_vectors, device, model_dtype
        )
        print(
            f"Ablating {directions_source} direction at all {num_layers_ablated} layers "
            f"via {ablation_method}.",
            flush=True,
        )
    elif ablation_method == ABLATION_METHOD_ORTHOGONALISATION:
        saved_weights = apply_weight_orthogonalisation(
            model, direction_vectors, device, model_dtype
        )
        print(
            f"Ablating {directions_source} direction at all {num_layers_ablated} layers "
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

    sampled_layer_indices = list(range(0, num_layers_ablated, 4))
    print_direction_norms(direction_vectors, sampled_layer_indices)

    probe_answers_by_index, probe_file_used = load_probe_answers_by_index(probe_file)

    dataset, question_count = load_forget_split_dataset(num_questions, FORGET_SPLIT)
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
                model,
                tokenizer,
                question,
                device,
                max_new_tokens,
                repetition_penalty=repetition_penalty,
            )
            progress.set_postfix_str(f"answered {index + 1}/{question_count}", refresh=True)

            result_entry = {
                "index": index,
                "question": question,
                "ground_truth": ground_truth_answer,
                "model_answer": model_answer,
                "rouge_l": compute_rouge_l(ground_truth_answer, model_answer),
            }
            if index in probe_answers_by_index:
                result_entry["probe_answer"] = probe_answers_by_index[index]
            results.append(result_entry)
    finally:
        if ablation_handles is not None:
            remove_ablation_hooks(ablation_handles)
        if saved_weights is not None:
            restore_original_weights(model, saved_weights)

    run_record = {
        "model": model_id,
        "model_key": model_key,
        "directions_file": directions_file,
        "directions_source": directions_source,
        "ablation_method": ablation_method,
        "steering_layer": steering_layer,
        "steering_coefficient": steering_coefficient,
        "repetition_penalty": repetition_penalty,
        "probe_file": probe_file_used,
        "num_layers_ablated": num_layers_ablated,
        "split": FORGET_SPLIT,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": question_count,
        "results": results,
    }
    attach_ablate_summaries_before_results(
        run_record,
        summarize_flat_results(results),
        load_probe_summary(probe_file_used, question_count),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(run_record, indent=2) + "\n")
        print(f"Saved results to {output_path}", flush=True)

    return run_record
