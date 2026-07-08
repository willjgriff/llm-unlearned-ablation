#!/usr/bin/env python3
"""
batch_npo_refusal_ablate.py - refusal direction + two-phase steering ablation.

For each selected NPO model:
  1. Extract refusal directions at 400 questions per split.
  2. Screen steering coefficients at a lower question count.
  3. Confirm the top 2 coefficients at 400 questions.

Setup:
    pip install -r requirements.txt

Usage:
    python src/batch_npo_refusal_ablate.py --skip-existing

    python src/batch_npo_refusal_ablate.py --screen-only --dry-run

    python src/batch_npo_refusal_ablate.py --confirm-only --skip-existing
"""
import argparse

from analysis.batch_npo_refusal_ablate import (
    DEFAULT_NPO_REFUSAL_ABLATE_SUMMARY_PATH,
    DEFAULT_STEERING_COEFFICIENTS,
    DEFAULT_TOP_NPO_REFUSAL_ABLATE_MODEL_KEYS,
    run_batch_npo_refusal_ablate,
)
from analysis.plot_direction_projection import DEFAULT_PROJECTION_PLOT_LAYER


def main():
    """Parse CLI arguments and run the NPO refusal ablation batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        action="append",
        default=None,
        help="model key from config/models.yaml (repeatable; default: top 2 beta0.1 NPO models)",
    )
    parser.add_argument(
        "--direction-num-questions",
        type=int,
        default=400,
        help="questions per split for refusal direction extraction",
    )
    parser.add_argument(
        "--screen-num-questions",
        type=int,
        default=50,
        help="forget-set questions per coefficient during screening sweep",
    )
    parser.add_argument(
        "--confirm-num-questions",
        type=int,
        default=400,
        help="forget-set questions for full confirm runs on top coefficients",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="maximum tokens to generate per question",
    )
    parser.add_argument(
        "--steering-layer",
        type=int,
        default=DEFAULT_PROJECTION_PLOT_LAYER,
        help="layer index for steering and projection plot",
    )
    parser.add_argument(
        "--steering-coefficients",
        nargs="+",
        type=float,
        default=None,
        help="coefficients to sweep during screening (default: 1.0 1.5 2.0 2.5 3.0 4.0)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="generation repetition penalty (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="number of top screening coefficients to confirm at full question count",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip steps whose output files already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned runs without loading models",
    )
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="run refusal direction extraction and screening sweeps only",
    )
    parser.add_argument(
        "--confirm-only",
        action="store_true",
        help="run full confirm runs from existing screening sweeps only",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="path to write batch summary JSON (default: results/ablate-and-probe/npo_refusal_ablate_summary.json)",
    )
    arguments = parser.parse_args()

    if arguments.screen_only and arguments.confirm_only:
        parser.error("Use at most one of --screen-only and --confirm-only.")

    run_screen = not arguments.confirm_only
    run_confirm = not arguments.screen_only
    model_keys = arguments.model_key or list(DEFAULT_TOP_NPO_REFUSAL_ABLATE_MODEL_KEYS)
    steering_coefficients = (
        arguments.steering_coefficients or list(DEFAULT_STEERING_COEFFICIENTS)
    )
    summary_output_path = arguments.summary_output or str(
        DEFAULT_NPO_REFUSAL_ABLATE_SUMMARY_PATH
    )

    print(f"Selected {len(model_keys)} models.", flush=True)
    run_batch_npo_refusal_ablate(
        model_keys=model_keys,
        direction_num_questions=arguments.direction_num_questions,
        screen_num_questions=arguments.screen_num_questions,
        confirm_num_questions=arguments.confirm_num_questions,
        max_new_tokens=arguments.max_new_tokens,
        steering_layer=arguments.steering_layer,
        steering_coefficients=steering_coefficients,
        repetition_penalty=arguments.repetition_penalty,
        top_k=arguments.top_k,
        skip_existing=arguments.skip_existing,
        dry_run=arguments.dry_run,
        run_screen=run_screen,
        run_confirm=run_confirm,
        summary_output_path=summary_output_path,
    )


if __name__ == "__main__":
    main()
