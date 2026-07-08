"""Batch TOFU forget-set probing for NPO unlearned checkpoints on Hugging Face."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from probing.behavioral import run_probe

PROBE_RESULTS_DIR = Path("results/probe")
DEFAULT_NPO_SCREENING_SUMMARY_PATH = PROBE_RESULTS_DIR / "npo_screening_summary.json"
HF_ORG = "open-unlearning"
DEFAULT_HF_SEARCH = "forget10_NPO_Llama-3.2-1B"
HF_MODEL_ID_PATTERN = re.compile(
    r"^open-unlearning/unlearn_tofu_Llama-3\.2-1B-Instruct_forget10_NPO_.+$"
)
PRIORITY_NPO_MODEL_KEYS = [
    "npo_unlearned_lr5e-05_beta0.5_alpha5_epoch10",
    "npo_unlearned_lr5e-05_beta0.5_alpha5_epoch5",
    "npo_unlearned_lr5e-05_beta0.5_alpha2_epoch10",
    "npo_unlearned_lr5e-05_beta0.1_alpha5_epoch10",
    "npo_unlearned_lr2e-05_beta0.1_alpha5_epoch10",
    "npo_unlearned_lr2e-05_beta0.05_alpha5_epoch10",
    "npo_unlearned_lr5e-05_beta0.1_alpha2_epoch5",
]


def hf_id_to_model_key(hf_id):
    """
    Convert an open-unlearning NPO Hugging Face repo id to a local model key.

    Args:
        hf_id: Full Hugging Face model id, e.g.
            open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha5_epoch5.

    Returns:
        Local model key string, e.g. npo_unlearned_lr2e-05_beta0.5_alpha5_epoch5.
    """
    repo_name = hf_id.split("/", 1)[-1]
    prefix = "unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_"
    if not repo_name.startswith(prefix):
        raise ValueError(f"Unexpected NPO Hugging Face repo name: {repo_name}")
    return "npo_unlearned_" + repo_name[len(prefix) :]


def model_key_to_probe_output_path(model_key):
    """
    Derive the default probe JSON output path for an NPO model key.

    Args:
        model_key: Local model key derived from a Hugging Face repo name.

    Returns:
        Path string under results/probe/.
    """
    return str(PROBE_RESULTS_DIR / f"{model_key}.json")


def discover_npo_forget10_hf_ids(search_term=DEFAULT_HF_SEARCH):
    """
    Discover NPO forget10 Llama-3.2-1B checkpoints from the Hugging Face API.

    Args:
        search_term: Search string passed to the Hugging Face models API.

    Returns:
        Sorted list of full Hugging Face model ids.
    """
    query = urlencode({"author": HF_ORG, "search": search_term, "limit": 100})
    api_url = f"https://huggingface.co/api/models?{query}"
    with urlopen(api_url) as response:
        models = json.load(response)

    discovered_ids = []
    for model_record in models:
        model_id = model_record["id"]
        if HF_MODEL_ID_PATTERN.match(model_id):
            discovered_ids.append(model_id)
    return sorted(discovered_ids)


def filter_priority_hf_ids(hf_ids):
    """
    Keep only priority NPO checkpoints from a Hugging Face id list.

    Args:
        hf_ids: List of full Hugging Face model ids.

    Returns:
        Sorted list of priority model ids.
    """
    priority_keys = set(PRIORITY_NPO_MODEL_KEYS)
    priority_ids = [
        hf_id for hf_id in hf_ids if hf_id_to_model_key(hf_id) in priority_keys
    ]
    return sorted(priority_ids)


def load_probe_summary_from_file(probe_path):
    """
    Read summary metrics from an existing probe JSON file.

    Args:
        probe_path: Path to a probe results JSON file.

    Returns:
        Dict with mean_rouge_l, count_above_0.3, and count_above_0.6.
    """
    probe_record = json.loads(Path(probe_path).read_text(encoding="utf-8"))
    summary = probe_record["summary"]
    return {
        "mean_rouge_l": summary["mean_rouge_l"],
        "count_above_0.3": summary["count_above_0.3"],
        "count_above_0.6": summary["count_above_0.6"],
    }


def build_screening_summary(model_entries, num_questions):
    """
    Build a leaderboard summary from per-model screening entries.

    Args:
        model_entries: List of dicts with model_key, hf_id, output_path, status,
            and optional summary metrics.
        num_questions: Number of forget-set questions used for probing.

    Returns:
        Summary dict sorted by count_above_0.3 ascending.
    """
    ranked_entries = [
        entry
        for entry in model_entries
        if entry.get("summary") is not None
    ]
    ranked_entries.sort(key=lambda entry: entry["summary"]["count_above_0.3"])

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": num_questions,
        "models": model_entries,
        "ranked_by_count_above_0.3": [
            {
                "model_key": entry["model_key"],
                "hf_id": entry["hf_id"],
                "count_above_0.3": entry["summary"]["count_above_0.3"],
                "mean_rouge_l": entry["summary"]["mean_rouge_l"],
            }
            for entry in ranked_entries
        ],
    }


def write_screening_summary(summary_record, summary_output_path):
    """
    Write the NPO screening leaderboard JSON to disk.

    Args:
        summary_record: Summary dict from build_screening_summary().
        summary_output_path: Destination JSON path.

    Returns:
        Path string for the written summary file.
    """
    summary_output_path = Path(summary_output_path)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(json.dumps(summary_record, indent=2) + "\n")
    print(f"Saved NPO screening summary to {summary_output_path}", flush=True)
    return str(summary_output_path)


def run_batch_npo_probe(
    hf_ids,
    num_questions,
    max_new_tokens,
    skip_existing,
    dry_run,
    summary_output_path,
):
    """
    Probe a list of NPO checkpoints on the TOFU forget10 split.

    Args:
        hf_ids: Hugging Face model ids to probe.
        num_questions: Number of forget-set questions to run per model.
        max_new_tokens: Maximum generated tokens per question.
        skip_existing: Skip models whose probe JSON output already exists.
        dry_run: Print planned runs without loading models or writing probe JSON.
        summary_output_path: Path to write the screening leaderboard JSON.

    Returns:
        Summary dict with per-model status and ROUGE metrics.
    """
    model_entries = []

    for index, hf_id in enumerate(hf_ids, start=1):
        model_key = hf_id_to_model_key(hf_id)
        output_path = model_key_to_probe_output_path(model_key)
        entry = {
            "model_key": model_key,
            "hf_id": hf_id,
            "output_path": output_path,
            "status": "pending",
            "summary": None,
        }

        if skip_existing and Path(output_path).exists():
            entry["status"] = "skipped_existing"
            entry["summary"] = load_probe_summary_from_file(output_path)
            print(
                f"[{index}/{len(hf_ids)}] Skipping existing probe for {model_key}",
                flush=True,
            )
            model_entries.append(entry)
            continue

        if dry_run:
            entry["status"] = "dry_run"
            print(
                f"[{index}/{len(hf_ids)}] Would probe {model_key} -> {output_path}",
                flush=True,
            )
            model_entries.append(entry)
            continue

        print(
            f"[{index}/{len(hf_ids)}] Probing {model_key} ({hf_id})...",
            flush=True,
        )
        try:
            run_record = run_probe(
                model_id=hf_id,
                num_questions=num_questions,
                max_new_tokens=max_new_tokens,
                output_path=output_path,
                model_key=model_key,
            )
            entry["status"] = "completed"
            entry["summary"] = run_record["summary"]
        except Exception as error:
            entry["status"] = "failed"
            entry["error"] = str(error)
            print(f"Failed to probe {model_key}: {error}", flush=True)

        model_entries.append(entry)

    summary_record = build_screening_summary(model_entries, num_questions)
    if not dry_run:
        write_screening_summary(summary_record, summary_output_path)
    return summary_record


def rebuild_screening_summary_from_existing_probes(
    num_questions,
    summary_output_path,
):
    """
    Rebuild the NPO screening leaderboard from existing probe JSON files.

    Args:
        num_questions: Number of forget-set questions assumed for the screen.
        summary_output_path: Path to write the screening leaderboard JSON.

    Returns:
        Summary dict built from results/probe/npo_unlearned_*.json files.
    """
    probe_paths = sorted(
        path
        for path in PROBE_RESULTS_DIR.glob("npo_unlearned_*.json")
        if not path.name.endswith("_high_rouge.json")
        and path.name != DEFAULT_NPO_SCREENING_SUMMARY_PATH.name
    )

    model_entries = []
    for probe_path in probe_paths:
        model_key = probe_path.stem
        probe_record = json.loads(probe_path.read_text(encoding="utf-8"))
        model_entries.append(
            {
                "model_key": model_key,
                "hf_id": probe_record.get("model", ""),
                "output_path": str(probe_path),
                "status": "loaded_existing",
                "summary": load_probe_summary_from_file(probe_path),
            }
        )

    summary_record = build_screening_summary(model_entries, num_questions)
    write_screening_summary(summary_record, summary_output_path)
    return summary_record
