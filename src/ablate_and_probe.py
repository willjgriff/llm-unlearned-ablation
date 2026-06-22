#!/usr/bin/env python3
"""
ablate_and_probe.py - ablate a saved direction and probe forget-set questions.

Loads per-layer directions from direction-calculation scripts, applies directional
intervention (hooks, weight orthogonalisation, or single-layer steering), then runs
forget-set questions and saves model answers alongside ground-truth to JSON.

Setup:
    pip install -r requirements.txt

Usage:
    python src/ablate_and_probe.py --model-key npo_unlearned
"""
import argparse

from probing.ablate_runner import ablate_and_probe
from utils.constants import (
    ABLATION_METHOD_HOOKS,
    ABLATION_METHOD_ORTHOGONALISATION,
    ABLATION_METHOD_STEER,
    DIRECTION_SOURCE_CONFABULATION,
    DIRECTION_SOURCE_REFUSAL,
)
from utils.model_config import get_model
from utils.paths import (
    default_ablate_and_probe_output_path,
    resolve_directions_file,
    resolve_probe_file,
)


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
    parser.add_argument(
        "--steering-coefficient",
        type=float,
        default=1.0,
        help="multiplier for the steering direction when --ablation-method steer (default: 1.0)",
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
        "--probe-file",
        default=None,
        help="path to probe JSON for non-ablated answers (default from config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to write JSON results (default from config; pass empty string to skip)",
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

    model_entry = get_model(arguments.model_key)
    directions_file = resolve_directions_file(
        model_entry,
        arguments.directions_file,
        arguments.directions_source,
    )
    output_path = arguments.output
    if output_path is None:
        output_path = default_ablate_and_probe_output_path(
            model_entry, arguments.directions_source
        )
    elif output_path == "":
        output_path = None

    probe_file = resolve_probe_file(model_entry, arguments.probe_file)

    ablate_and_probe(
        model_id=model_entry["hf_id"],
        directions_file=directions_file,
        num_questions=arguments.num_questions,
        max_new_tokens=arguments.max_new_tokens,
        ablation_method=arguments.ablation_method,
        directions_source=arguments.directions_source,
        steering_layer=arguments.steering_layer,
        steering_coefficient=arguments.steering_coefficient,
        probe_file=probe_file,
        output_path=output_path,
        model_key=arguments.model_key,
    )


if __name__ == "__main__":
    main()
