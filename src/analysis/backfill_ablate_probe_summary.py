"""Backfill probe_summary into existing ablate-and-probe result files."""

import json
from pathlib import Path

from utils.paths import ABLATE_AND_PROBE_RESULTS_DIR, load_probe_summary

ABLATE_RESULT_SUMMARY_KEYS = ("mean_rouge_l", "count_above_0.3", "count_above_0.6")


def is_ablate_result_file(path):
    """
    Return True when a JSON path looks like an ablate-and-probe results file.

    Args:
        path: Path to inspect.

    Returns:
        True for standard ablate result files, excluding extractor outputs.
    """
    path = Path(path)
    if path.suffix != ".json":
        return False
    if path.stem.endswith("_high_rouge"):
        return False
    if path.name.endswith("_summary.json"):
        return False
    return True


def has_rouge_summary(summary):
    """
    Return True when a summary block contains ablation ROUGE metrics.

    Args:
        summary: Parsed summary dict from an ablate-and-probe JSON file.

    Returns:
        True if the summary has flat or sweep ROUGE metrics.
    """
    if any(key in summary for key in ABLATE_RESULT_SUMMARY_KEYS):
        return True
    return "per_coefficient" in summary or "across_coefficients" in summary


def resolve_ablate_result_paths(input_dir=None):
    """
    Resolve ablate-and-probe JSON files that may need probe_summary backfill.

    Args:
        input_dir: Optional directory to scan instead of results/ablate-and-probe.

    Returns:
        Sorted list of ablate result file paths.
    """
    scan_directory = Path(input_dir or ABLATE_AND_PROBE_RESULTS_DIR)
    return sorted(
        path
        for path in scan_directory.rglob("*.json")
        if is_ablate_result_file(path)
    )


def backfill_probe_summary_in_ablate_file(ablate_result_path):
    """
    Add or refresh probe_summary in one ablate-and-probe JSON file.

    Args:
        ablate_result_path: Path to an ablate-and-probe results JSON file.

    Returns:
        Status string: updated, skipped_no_summary, skipped_no_probe, or skipped_missing_probe_file.
    """
    ablate_result_path = Path(ablate_result_path)
    ablate_record = json.loads(ablate_result_path.read_text(encoding="utf-8"))

    if "summary" not in ablate_record or "results" not in ablate_record:
        return "skipped_no_summary"

    if not has_rouge_summary(ablate_record["summary"]):
        return "skipped_no_summary"

    probe_file = ablate_record.get("probe_file")
    num_questions = ablate_record.get("num_questions")
    if not probe_file or num_questions is None:
        return "skipped_no_probe"

    probe_summary = load_probe_summary(probe_file, num_questions)
    if probe_summary is None:
        return "skipped_missing_probe_file"

    results = ablate_record.pop("results")
    ablate_record["probe_summary"] = probe_summary
    ablate_record["results"] = results
    ablate_result_path.write_text(json.dumps(ablate_record, indent=2) + "\n")
    return "updated"


def backfill_probe_summary_in_ablate_files(input_dir=None):
    """
    Backfill probe_summary across all matching ablate-and-probe result files.

    Args:
        input_dir: Optional directory to scan instead of results/ablate-and-probe.

    Returns:
        Dict mapping status strings to lists of updated file paths.
    """
    status_to_paths = {
        "updated": [],
        "skipped_no_summary": [],
        "skipped_no_probe": [],
        "skipped_missing_probe_file": [],
    }

    for ablate_result_path in resolve_ablate_result_paths(input_dir=input_dir):
        status = backfill_probe_summary_in_ablate_file(ablate_result_path)
        status_to_paths[status].append(str(ablate_result_path))

    return status_to_paths
