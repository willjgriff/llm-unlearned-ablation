"""Extract high-ROUGE ablate-and-probe responses for manual verification."""

import json
from pathlib import Path

ROUGE_LOW_THRESHOLD = 0.3
ROUGE_HIGH_THRESHOLD = 0.6


def default_high_rouge_output_path(input_path):
    """
    Derive the default output path for a high-ROUGE extraction file.

    Args:
        input_path: Path to the source ablate-and-probe JSON file.

    Returns:
        Output path string with _high_rouge.json suffix.
    """
    input_file_path = Path(input_path)
    return str(input_file_path.with_name(input_file_path.stem + "_high_rouge.json"))


def build_verification_entry(
    result_entry,
    model_answer,
    rouge_l,
    steering_coefficient=None,
):
    """
    Build one verification entry with model_answer immediately before rouge_l.

    Args:
        result_entry: Per-question dict from an ablate-and-probe JSON file.
        model_answer: Ablated model answer text.
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
    Expand ablate-and-probe results into flat rows for filtering.

    Sweep entries with model_answers produce one row per coefficient. Flat
    entries with model_answer produce a single row.

    Args:
        results: List of per-question dicts from an ablate-and-probe JSON file.

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
    Build a high-ROUGE verification record from an ablate-and-probe JSON file.

    Args:
        input_record: Parsed top-level ablate-and-probe JSON object.
        input_path: Path to the source file (stored in output metadata).

    Returns:
        Dict with source metadata and above_0.6 / above_0.3 result sections.
    """
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
    Write high-ROUGE ablate-and-probe responses to JSON for manual verification.

    Args:
        input_path: Path to a source ablate-and-probe JSON file.
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
        f"Wrote {output_record['above_0.6']['count']} entries above 0.6 and "
        f"{output_record['above_0.3']['count']} entries between 0.3 and 0.6 "
        f"to {output_path}",
        flush=True,
    )
    return output_record
