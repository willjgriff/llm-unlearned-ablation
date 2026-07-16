#!/usr/bin/env python3
"""
ablate_and_probe.py - ablate a saved refusal direction and probe forget-set questions.

Loads per-layer directions from direction_refusal.py, applies directional ablation
at every layer (via forward hooks or weight orthogonalisation), then runs forget-set
questions and saves model answers alongside ground-truth to JSON.

Setup:
    pip install -r requirements.txt

Usage:
    python src/ablate_and_probe.py --model-key npo_unlearned
"""
import argparse

from model_config import get_model
from probing.ablate_runner import ablate_and_probe, run_coefficient_sweep
from utils.constants import (
    ABLATION_METHOD_HOOKS,
    ABLATION_METHOD_ORTHOGONALISATION,
    ABLATION_METHOD_STEER,
    DIRECTION_SOURCE_CONFABULATION,
    DIRECTION_SOURCE_REFUSAL,
    QUESTION_MODE_ORIGINAL,
    QUESTION_MODE_PERTURBED,
)
from utils.paths import (
    append_coefficient_suffix,
    append_layer_suffix,
    append_perturbed_suffix,
    build_ablate_probe_output_path,
    resolve_directions_file,
    resolve_probe_file,
)


def resolve_single_steering_coefficient(arguments):
    """
    Return the steering coefficient when exactly one value is in use.

    Args:
        arguments: Parsed CLI namespace from ablate_and_probe.py.

    Returns:
        Single steering coefficient float, or None for sweeps and non-steer runs.
    """
    if arguments.steering_coefficients is not None:
        if len(arguments.steering_coefficients) == 1:
            return arguments.steering_coefficients[0]
        return None
    if arguments.ablation_method == ABLATION_METHOD_STEER:
        return arguments.steering_coefficient
    return None


def main():
    """Parse CLI arguments and run ablation probing."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        required=True,
        help="short name from config/models.yaml (e.g. npo_unlearned)",
    )
    parser.add_argument(
        "--directions-file",
        default=None,
        help="path to directions .pt file (overrides --directions-source)",
    )
    parser.add_argument(
        "--directions-source",
        choices=[DIRECTION_SOURCE_REFUSAL, DIRECTION_SOURCE_CONFABULATION],
        default=DIRECTION_SOURCE_REFUSAL,
        help="which saved direction file to load from config (default: refusal)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ablation-method",
        choices=[
            ABLATION_METHOD_HOOKS,
            ABLATION_METHOD_ORTHOGONALISATION,
            ABLATION_METHOD_STEER,
        ],
        default=ABLATION_METHOD_HOOKS,
        help="how to apply directional intervention (default: hooks)",
    )
    steering_coefficient_group = parser.add_mutually_exclusive_group()
    steering_coefficient_group.add_argument(
        "--steering-coefficient",
        type=float,
        default=1.0,
        help="multiplier for the steering direction when --ablation-method steer (default: 1.0)",
    )
    steering_coefficient_group.add_argument(
        "--steering-coefficients",
        nargs="+",
        type=float,
        help="space-separated steering coefficients for a sweep (requires --ablation-method steer)",
    )
    parser.add_argument(
        "--steering-layer",
        type=int,
        default=None,
        help="layer index for single-layer steering (required when --ablation-method steer)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="number of questions to probe",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="penalty for repeating tokens during generation (default: 1.0)",
    )
    parser.add_argument(
        "--probe-file",
        default=None,
        help="path to probe JSON for non-ablated answers (default from config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to write JSON results (default from config; pass empty string to skip)",
    )
    parser.add_argument(
        "--question-mode",
        choices=[QUESTION_MODE_ORIGINAL, QUESTION_MODE_PERTURBED],
        default=QUESTION_MODE_ORIGINAL,
        help="use original forget10 questions or paraphrased forget10_perturbed prompts",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="optional TOFU config override (default: forget10 or forget10_perturbed)",
    )
    arguments = parser.parse_args()

    if arguments.layer is not None:
        print(
            "Warning: --layer is ignored; use --steering-layer with --ablation-method steer.",
            flush=True,
        )

    if arguments.ablation_method == ABLATION_METHOD_STEER:
        if arguments.steering_layer is None:
            parser.error("--steering-layer is required when --ablation-method is steer")

    if arguments.steering_coefficients is not None:
        if arguments.ablation_method != ABLATION_METHOD_STEER:
            parser.error("--steering-coefficients requires --ablation-method steer")

    model_entry = get_model(arguments.model_key)
    directions_file = resolve_directions_file(
        model_entry,
        arguments.directions_file,
        arguments.directions_source,
    )
    output_path = arguments.output
    if output_path is None:
        is_coefficient_sweep = (
            arguments.steering_coefficients is not None
            and len(arguments.steering_coefficients) > 1
        )
        uses_steering = (
            arguments.ablation_method == ABLATION_METHOD_STEER
            or arguments.steering_coefficients is not None
        )
        output_path = build_ablate_probe_output_path(
            arguments.model_key,
            arguments.directions_source,
            ablation_method=arguments.ablation_method,
            is_coefficient_sweep=is_coefficient_sweep,
            steering_layer=arguments.steering_layer if uses_steering else None,
            steering_coefficient=resolve_single_steering_coefficient(arguments),
        )
    elif output_path == "":
        output_path = None

    if output_path is not None and arguments.question_mode == QUESTION_MODE_PERTURBED:
        output_path = append_perturbed_suffix(output_path)

    uses_steering = (
        arguments.ablation_method == ABLATION_METHOD_STEER
        or arguments.steering_coefficients is not None
    )
    if arguments.output is not None and arguments.output != "" and uses_steering:
        output_path = append_layer_suffix(output_path, arguments.steering_layer)
        single_steering_coefficient = resolve_single_steering_coefficient(arguments)
        if single_steering_coefficient is not None:
            output_path = append_coefficient_suffix(
                output_path, single_steering_coefficient
            )

    probe_file = resolve_probe_file(
        model_entry, arguments.probe_file, arguments.question_mode
    )

    shared_probe_kwargs = {
        "question_mode": arguments.question_mode,
        "split_name": arguments.split,
    }

    if arguments.steering_coefficients is not None:
        run_coefficient_sweep(
            model_id=model_entry["hf_id"],
            directions_file=directions_file,
            num_questions=arguments.num_questions,
            max_new_tokens=arguments.max_new_tokens,
            directions_source=arguments.directions_source,
            steering_layer=arguments.steering_layer,
            steering_coefficients=arguments.steering_coefficients,
            repetition_penalty=arguments.repetition_penalty,
            probe_file=probe_file,
            output_path=output_path,
            model_key=arguments.model_key,
            **shared_probe_kwargs,
        )
    else:
        ablate_and_probe(
            model_id=model_entry["hf_id"],
            directions_file=directions_file,
            num_questions=arguments.num_questions,
            max_new_tokens=arguments.max_new_tokens,
            ablation_method=arguments.ablation_method,
            directions_source=arguments.directions_source,
            steering_layer=arguments.steering_layer,
            steering_coefficient=arguments.steering_coefficient,
            repetition_penalty=arguments.repetition_penalty,
            probe_file=probe_file,
            output_path=output_path,
            model_key=arguments.model_key,
            **shared_probe_kwargs,
        )


if __name__ == "__main__":
    main()
