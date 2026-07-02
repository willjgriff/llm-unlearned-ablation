"""Extract high-ROUGE probe and ablate-and-probe responses for manual verification."""

import json
from pathlib import Path

ROUGE_LOW_THRESHOLD = 0.3
ROUGE_HIGH_THRESHOLD = 0.6


def default_high_rouge_output_path(input_path):
    """
    Derive the default output path for a high-ROUGE extraction file.

    Args:
        input_path: Path to the source probe or ablate-and-probe JSON file.

    Returns:
        Output path string with _high_rouge.json suffix.
    """
    input_file_path = Path(input_path)
    return str(input_file_path.with_name(input_file_path.stem + "_high_rouge.json"))


def is_high_rouge_output_path(path):
    """
    Return True when a path looks like an extractor output file.

    Args:
        path: File path to inspect.

    Returns:
        True if the filename stem ends with _high_rouge.
    """
    return Path(path).stem.endswith("_high_rouge")


def infer_source_type(input_record):
    """
    Infer whether a results file came from behavioral probe or ablate-and-probe.

    Args:
        input_record: Parsed top-level JSON object.

    Returns:
        Either "probe" or "ablate_and_probe".
    """
    if "ablation_method" in input_record or "directions_source" in input_record:
        return "ablate_and_probe"
    return "probe"


def resolve_input_paths(input_paths=None, input_dir=None):
    """
    Resolve probe or ablate-and-probe JSON files to extract from.

    Skips files that already look like high-ROUGE extractor outputs.

    Args:
        input_paths: Optional list of explicit file paths or glob patterns.
        input_dir: Optional directory to scan for *.json source files.

    Returns:
        Sorted list of unique input file paths.

    Raises:
        ValueError: If no inputs are provided or no matching files are found.
    """
    resolved_paths = []

    if input_paths:
        for input_path in input_paths:
            path = Path(input_path)
            if any(char in input_path for char in "*?[]"):
                resolved_paths.extend(sorted(path.parent.glob(path.name)))
            elif path.is_dir():
                resolved_paths.extend(sorted(path.glob("*.json")))
            elif path.is_file():
                resolved_paths.append(path)
            else:
                raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_dir:
        input_directory = Path(input_dir)
        if not input_directory.is_dir():
            raise NotADirectoryError(f"Input directory not found: {input_dir}")
        resolved_paths.extend(sorted(input_directory.glob("*.json")))

    if not input_paths and not input_dir:
        raise ValueError("Provide at least one input file or --input-dir.")

    unique_paths = []
    seen_paths = set()
    for path in resolved_paths:
        normalized_path = path.resolve()
        if normalized_path in seen_paths:
            continue
        if is_high_rouge_output_path(path):
            continue
        seen_paths.add(normalized_path)
        unique_paths.append(path)

    if not unique_paths:
        raise ValueError("No source JSON files found to extract.")

    return unique_paths


def build_verification_entry(
    result_entry,
    model_answer,
    rouge_l,
    steering_coefficient=None,
):
    """
    Build one verification entry with model_answer immediately before rouge_l.

    Args:
        result_entry: Per-question dict from a probe or ablate-and-probe JSON file.
        model_answer: Model answer text.
        rouge_l: ROUGE-L score for this answer against ground_truth.
        steering_coefficient: Optional steering coefficient for sweep rows.

    Returns:
        Ordered dict for manual review.
    """
    verification_entry = {
        "index": result_entry["index"],
        "question": result_entry["question"],
        "ground_truth": result_entry["ground_truth"],
    }
    if "probe_answer" in result_entry:
        verification_entry["probe_answer"] = result_entry["probe_answer"]
    if steering_coefficient is not None:
        verification_entry["steering_coefficient"] = steering_coefficient
    verification_entry["model_answer"] = model_answer
    verification_entry["rouge_l"] = rouge_l
    return verification_entry


def expand_result_entries(results):
    """
    Expand probe or ablate-and-probe results into flat rows for filtering.

    Sweep entries with model_answers produce one row per coefficient. Flat
    entries with model_answer produce a single row.

    Args:
        results: List of per-question dicts from a probe or ablate-and-probe file.

    Returns:
        List of verification entry dicts.
    """
    expanded_entries = []
    for result_entry in results:
        if "model_answers" in result_entry:
            for coefficient_answer in result_entry["model_answers"]:
                expanded_entries.append(
                    build_verification_entry(
                        result_entry,
                        coefficient_answer["model_answer"],
                        coefficient_answer["rouge_l"],
                        steering_coefficient=coefficient_answer["steering_coefficient"],
                    )
                )
            continue

        expanded_entries.append(
            build_verification_entry(
                result_entry,
                result_entry["model_answer"],
                result_entry["rouge_l"],
            )
        )
    return expanded_entries


def filter_verification_entries(expanded_entries):
    """
    Split expanded entries into mutually exclusive ROUGE bands.

    Args:
        expanded_entries: Flat verification entry dicts with rouge_l scores.

    Returns:
        Tuple of (above_0.6 entries, above_0.3_up_to_0.6 entries).
    """
    above_0_6_entries = []
    above_0_3_up_to_0_6_entries = []

    for entry in expanded_entries:
        rouge_l = entry["rouge_l"]
        if rouge_l > ROUGE_HIGH_THRESHOLD:
            above_0_6_entries.append(entry)
        elif rouge_l > ROUGE_LOW_THRESHOLD:
            above_0_3_up_to_0_6_entries.append(entry)

    sort_key = lambda entry: entry["rouge_l"]
    above_0_6_entries.sort(key=sort_key, reverse=True)
    above_0_3_up_to_0_6_entries.sort(key=sort_key, reverse=True)
    return above_0_6_entries, above_0_3_up_to_0_6_entries


def build_high_rouge_record(input_record, input_path):
    """
    Build a high-ROUGE verification record from a probe or ablate-and-probe file.

    Args:
        input_record: Parsed top-level JSON object.
        input_path: Path to the source file (stored in output metadata).

    Returns:
        Dict with source metadata and above_0.6 / above_0.3 result sections.

    Raises:
        ValueError: If the input record is missing a top-level results list.
    """
    if "results" not in input_record:
        raise ValueError("Expected top-level 'results' list in input JSON.")

    expanded_entries = expand_result_entries(input_record["results"])
    above_0_6_entries, above_0_3_up_to_0_6_entries = filter_verification_entries(
        expanded_entries
    )

    metadata = {
        key: value
        for key, value in input_record.items()
        if key not in {"results", "summary"}
    }
    metadata["source"] = str(input_path)
    metadata["source_type"] = infer_source_type(input_record)

    return {
        **metadata,
        "above_0.6": {
            "count": len(above_0_6_entries),
            "results": above_0_6_entries,
        },
        "above_0.3": {
            "count": len(above_0_3_up_to_0_6_entries),
            "results": above_0_3_up_to_0_6_entries,
        },
    }


def extract_high_rouge_responses(input_path, output_path):
    """
    Write high-ROUGE responses to JSON for manual verification.

    Args:
        input_path: Path to a source probe or ablate-and-probe JSON file.
        output_path: Path to write the filtered verification JSON.

    Returns:
        The output record dict that was written.
    """
    input_path = Path(input_path)
    input_record = json.loads(input_path.read_text(encoding="utf-8"))
    output_record = build_high_rouge_record(input_record, input_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_record, indent=2) + "\n")
    print(
        f"{input_path.name}: wrote {output_record['above_0.6']['count']} entries above 0.6 "
        f"and {output_record['above_0.3']['count']} entries between 0.3 and 0.6 "
        f"to {output_path}",
        flush=True,
    )
    return output_record


def extract_high_rouge_responses_batch(input_paths, output_path=None):
    """
    Extract high-ROUGE responses from one or more source JSON files.

    Args:
        input_paths: List of probe or ablate-and-probe JSON file paths.
        output_path: Optional explicit output path; only valid for a single input.

    Returns:
        List of output record dicts that were written.

    Raises:
        ValueError: If output_path is provided alongside multiple inputs.
    """
    if output_path is not None and len(input_paths) != 1:
        raise ValueError("--output requires exactly one input file.")

    output_records = []
    for input_path in input_paths:
        destination_path = output_path or default_high_rouge_output_path(input_path)
        output_records.append(
            extract_high_rouge_responses(input_path, destination_path)
        )
    return output_records
