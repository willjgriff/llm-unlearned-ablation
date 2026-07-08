#!/usr/bin/env python3
"""
batch_npo_probe.py - screen NPO unlearned checkpoints on the TOFU forget10 split.

Discovers open-unlearning NPO forget10 Llama-3.2-1B checkpoints on Hugging Face,
runs the behavioral probe on each, and writes a leaderboard sorted by
count_above_0.3.

Setup:
    pip install -r requirements.txt

Usage:
    python src/batch_npo_probe.py --priority-only --skip-existing

    python src/batch_npo_probe.py --dry-run

    python src/batch_npo_probe.py --summary-only
"""
import argparse

from analysis.batch_npo_probe import (
    DEFAULT_NPO_SCREENING_SUMMARY_PATH,
    discover_npo_forget10_hf_ids,
    filter_priority_hf_ids,
    rebuild_screening_summary_from_existing_probes,
    run_batch_npo_probe,
)


def main():
    """Parse CLI arguments and run the NPO checkpoint screening batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-id",
        action="append",
        default=None,
        help="explicit Hugging Face model id to probe (repeatable)",
    )
    parser.add_argument(
        "--discover-from-hf",
        action="store_true",
        help="discover NPO forget10 Llama-3.2-1B checkpoints from Hugging Face",
    )
    parser.add_argument(
        "--priority-only",
        action="store_true",
        help="probe only the priority hyperparameter subset",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=400,
        help="number of forget10 questions to probe per model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="maximum tokens to generate per question",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip models whose probe JSON already exists in results/probe/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned probe runs without loading models",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="rebuild the screening leaderboard from existing probe JSON files",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="path to write screening leaderboard JSON (default: results/probe/npo_screening_summary.json)",
    )
    arguments = parser.parse_args()

    summary_output_path = arguments.summary_output or str(
        DEFAULT_NPO_SCREENING_SUMMARY_PATH
    )

    if arguments.summary_only:
        rebuild_screening_summary_from_existing_probes(
            num_questions=arguments.num_questions,
            summary_output_path=summary_output_path,
        )
        return

    if arguments.hf_id:
        hf_ids = sorted(arguments.hf_id)
    elif arguments.discover_from_hf or not arguments.hf_id:
        hf_ids = discover_npo_forget10_hf_ids()
    else:
        raise ValueError("Provide --hf-id values or use --discover-from-hf.")

    if arguments.priority_only:
        hf_ids = filter_priority_hf_ids(hf_ids)

    if not hf_ids:
        raise ValueError("No NPO checkpoints matched the requested filters.")

    print(f"Selected {len(hf_ids)} NPO checkpoints.", flush=True)
    run_batch_npo_probe(
        hf_ids=hf_ids,
        num_questions=arguments.num_questions,
        max_new_tokens=arguments.max_new_tokens,
        skip_existing=arguments.skip_existing,
        dry_run=arguments.dry_run,
        summary_output_path=summary_output_path,
    )


if __name__ == "__main__":
    main()
