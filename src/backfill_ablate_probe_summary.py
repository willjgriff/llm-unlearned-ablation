#!/usr/bin/env python3
"""
backfill_ablate_probe_summary.py - add probe_summary to ablate-and-probe JSON files.

Adds the non-ablated probe ROUGE summary below each ablated summary block for
quick comparison when a probe_file is available.

Usage:
    python src/backfill_ablate_probe_summary.py

    python src/backfill_ablate_probe_summary.py --input-dir results/ablate-and-probe
"""
import argparse

from analysis.backfill_ablate_probe_summary import backfill_probe_summary_in_ablate_files


def main():
    """Parse CLI arguments and backfill probe_summary into ablate result files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=None,
        help="directory of ablate-and-probe JSON files to update",
    )
    arguments = parser.parse_args()

    status_to_paths = backfill_probe_summary_in_ablate_files(
        input_dir=arguments.input_dir
    )
    updated_paths = status_to_paths["updated"]
    print(f"Updated {len(updated_paths)} ablate result files.", flush=True)
    for path in updated_paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
