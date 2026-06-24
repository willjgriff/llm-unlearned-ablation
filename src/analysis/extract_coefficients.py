"""Filter and flatten ablate-and-probe JSON results by steering coefficient."""

import json
from pathlib import Path


def format_coefficient_for_key(coefficient):
    """
    Format a steering coefficient for use in a model_answers_coefficient key.

    Args:
        coefficient: Steering coefficient float value.

    Returns:
        String suffix for the key (e.g. 1.5 -> "1.5", 2.0 -> "2.0").
    """
    coefficient_float = float(coefficient)
    if coefficient_float == int(coefficient_float):
        return f"{coefficient_float:.1f}"
    return str(coefficient_float)


def coefficient_answer_key(coefficient):
    """
    Build the collapsed answer field name for a steering coefficient.

    Args:
        coefficient: Steering coefficient float value.

    Returns:
        Field name such as model_answers_coefficient1.5.
    """
    return f"model_answers_coefficient{format_coefficient_for_key(coefficient)}"


def coefficients_match(left_coefficient, right_coefficient):
    """
    Compare two steering coefficient values for equality.

    Args:
        left_coefficient: First coefficient value.
        right_coefficient: Second coefficient value.

    Returns:
        True when both values represent the same coefficient.
    """
    return abs(float(left_coefficient) - float(right_coefficient)) < 1e-9


def coefficient_is_requested(coefficient, requested_coefficients):
    """
    Check whether a coefficient appears in the requested filter list.

    Args:
        coefficient: Coefficient value from an input result entry.
        requested_coefficients: List of coefficient values to keep.

    Returns:
        True when the coefficient should be included in the output.
    """
    return any(
        coefficients_match(coefficient, requested_coefficient)
        for requested_coefficient in requested_coefficients
    )


def find_requested_coefficient(coefficient, requested_coefficients):
    """
    Return the matching requested coefficient value for stable key ordering.

    Args:
        coefficient: Coefficient value from an input result entry.
        requested_coefficients: List of coefficient values to keep.

    Returns:
        Matching value from requested_coefficients, or None if not requested.
    """
    for requested_coefficient in requested_coefficients:
        if coefficients_match(coefficient, requested_coefficient):
            return requested_coefficient
    return None


def extract_answers_by_coefficient(result_entry, requested_coefficients):
    """
    Extract model answers for requested coefficients from one result entry.

    Supports sweep entries with model_answers lists and single-coefficient entries
    with a top-level model_answer plus file-level steering_coefficient metadata.

    Args:
        result_entry: One per-question dict from an ablate-and-probe JSON file.
        requested_coefficients: List of coefficient values to keep.

    Returns:
        Dict mapping model_answers_coefficient keys to answer strings.
    """
    answers_by_key = {}

    if "model_answers" in result_entry:
        for answer_entry in result_entry["model_answers"]:
            matched_coefficient = find_requested_coefficient(
                answer_entry["steering_coefficient"], requested_coefficients
            )
            if matched_coefficient is not None:
                answers_by_key[coefficient_answer_key(matched_coefficient)] = (
                    answer_entry["model_answer"]
                )
        return answers_by_key

    file_steering_coefficient = result_entry.get("_file_steering_coefficient")
    if (
        "model_answer" in result_entry
        and file_steering_coefficient is not None
        and coefficient_is_requested(file_steering_coefficient, requested_coefficients)
    ):
        matched_coefficient = find_requested_coefficient(
            file_steering_coefficient, requested_coefficients
        )
        answers_by_key[coefficient_answer_key(matched_coefficient)] = result_entry[
            "model_answer"
        ]

    return answers_by_key


def collapse_result_entry(result_entry, requested_coefficients):
    """
    Build one output result entry with collapsed coefficient answer keys.

    Args:
        result_entry: One per-question dict from an ablate-and-probe JSON file.
        requested_coefficients: List of coefficient values to keep.

    Returns:
        Collapsed per-question dict with index, question, ground_truth, optional
        probe_answer, and model_answers_coefficient keys for each match.
    """
    collapsed_entry = {
        "index": result_entry["index"],
        "question": result_entry["question"],
        "ground_truth": result_entry["ground_truth"],
    }
    if "probe_answer" in result_entry:
        collapsed_entry["probe_answer"] = result_entry["probe_answer"]

    collapsed_entry.update(
        extract_answers_by_coefficient(result_entry, requested_coefficients)
    )
    return collapsed_entry


def resolve_available_coefficients(input_record):
    """
    Collect all steering coefficients present in an ablate-and-probe JSON file.

    Args:
        input_record: Parsed top-level ablate-and-probe JSON object.

    Returns:
        List of coefficient floats found in file metadata or per-question answers.
    """
    if "steering_coefficients" in input_record:
        return [float(coefficient) for coefficient in input_record["steering_coefficients"]]
    if "steering_coefficient" in input_record:
        return [float(input_record["steering_coefficient"])]

    discovered_coefficients = []
    for result_entry in input_record.get("results", []):
        if "model_answers" in result_entry:
            for answer_entry in result_entry["model_answers"]:
                coefficient = float(answer_entry["steering_coefficient"])
                if not any(
                    coefficients_match(coefficient, existing)
                    for existing in discovered_coefficients
                ):
                    discovered_coefficients.append(coefficient)
    return discovered_coefficients


def validate_requested_coefficients(requested_coefficients, available_coefficients):
    """
    Ensure every requested coefficient exists in the input file.

    Args:
        requested_coefficients: Coefficients requested on the command line.
        available_coefficients: Coefficients present in the input file.

    Raises:
        ValueError: When a requested coefficient is missing from the input file.
    """
    for requested_coefficient in requested_coefficients:
        if not coefficient_is_requested(requested_coefficient, available_coefficients):
            available_text = ", ".join(
                format_coefficient_for_key(coefficient)
                for coefficient in available_coefficients
            )
            raise ValueError(
                f"Coefficient {format_coefficient_for_key(requested_coefficient)} "
                f"not found in input file. Available: {available_text}"
            )


def prepare_result_entries_for_extraction(input_record):
    """
    Attach file-level steering metadata to single-coefficient result entries.

    Args:
        input_record: Parsed top-level ablate-and-probe JSON object.

    Returns:
        List of per-question dicts ready for coefficient extraction.
    """
    file_steering_coefficient = input_record.get("steering_coefficient")
    prepared_results = []
    for result_entry in input_record.get("results", []):
        prepared_entry = dict(result_entry)
        if file_steering_coefficient is not None and "model_answers" not in prepared_entry:
            prepared_entry["_file_steering_coefficient"] = file_steering_coefficient
        prepared_results.append(prepared_entry)
    return prepared_results


def build_output_record(input_record, requested_coefficients, collapsed_results):
    """
    Build the filtered output record using the input file's top-level schema.

    Args:
        input_record: Parsed top-level ablate-and-probe JSON object.
        requested_coefficients: Coefficient values to keep, in request order.
        collapsed_results: Transformed per-question result entries.

    Returns:
        Output dict with updated metadata and collapsed results.
    """
    output_record = {
        key: value
        for key, value in input_record.items()
        if key not in {"results", "steering_coefficient", "steering_coefficients"}
    }
    output_record["steering_coefficients"] = [
        float(coefficient) for coefficient in requested_coefficients
    ]
    output_record["results"] = collapsed_results
    return output_record


def default_output_path(input_path):
    """
    Derive a default output path by inserting _filtered before the file extension.

    Args:
        input_path: Path to the source ablate-and-probe JSON file.

    Returns:
        Default output path string.
    """
    input_file_path = Path(input_path)
    return str(
        input_file_path.with_name(
            input_file_path.stem + "_filtered" + input_file_path.suffix
        )
    )


def extract_ablate_probe_coefficients(input_path, requested_coefficients, output_path):
    """
    Filter and flatten ablate-and-probe results to selected steering coefficients.

    Args:
        input_path: Path to a source ablate-and-probe JSON file.
        requested_coefficients: List of steering coefficient values to keep.
        output_path: Path to write the filtered JSON file.

    Returns:
        Parsed output record written to disk.
    """
    input_file_path = Path(input_path)
    input_record = json.loads(input_file_path.read_text(encoding="utf-8"))

    available_coefficients = resolve_available_coefficients(input_record)
    validate_requested_coefficients(requested_coefficients, available_coefficients)

    prepared_results = prepare_result_entries_for_extraction(input_record)
    collapsed_results = [
        collapse_result_entry(result_entry, requested_coefficients)
        for result_entry in prepared_results
    ]

    output_record = build_output_record(
        input_record, requested_coefficients, collapsed_results
    )

    output_file_path = Path(output_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.write_text(json.dumps(output_record, indent=2) + "\n")
    print(f"Saved filtered results to {output_file_path}", flush=True)

    return output_record
