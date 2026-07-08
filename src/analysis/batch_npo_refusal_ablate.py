"""Two-phase refusal-direction extraction and steering ablation for NPO models."""

import json
from datetime import datetime, timezone
from pathlib import Path

from analysis.plot_direction_projection import DEFAULT_PROJECTION_PLOT_LAYER
from direction_calculation.direction_refusal import extract_refusal_directions
from model_config import get_model
from probing.ablate_runner import ablate_and_probe, run_coefficient_sweep
from utils.constants import ABLATION_METHOD_STEER, DIRECTION_SOURCE_REFUSAL
from utils.paths import (
    ABLATE_AND_PROBE_RESULTS_DIR,
    build_ablate_probe_output_path,
    default_refusal_direction_projection_plot_path,
)

DEFAULT_NPO_REFUSAL_ABLATE_SUMMARY_PATH = (
    ABLATE_AND_PROBE_RESULTS_DIR / "npo_refusal_ablate_summary.json"
)
DEFAULT_TOP_NPO_REFUSAL_ABLATE_MODEL_KEYS = [
    "npo_unlearned_lr5e-05_beta0.1_alpha5_epoch10",
    "npo_unlearned_lr5e-05_beta0.1_alpha2_epoch5",
]
DEFAULT_STEERING_COEFFICIENTS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def default_screen_sweep_output_path(model_key, steering_layer, screen_num_questions):
    """
    Derive the output path for a low-question steering coefficient sweep.

    Args:
        model_key: Config key for the model.
        steering_layer: Layer index used for steering.
        screen_num_questions: Number of forget-set questions used in the sweep.

    Returns:
        Path string such as negsteer_sweep_layer14_n50_refusal.json.
    """
    filename = (
        f"negsteer_sweep_layer{steering_layer}_n{screen_num_questions}_refusal.json"
    )
    return str(ABLATE_AND_PROBE_RESULTS_DIR / model_key / filename)


def default_confirm_steering_output_path(model_key, steering_layer, steering_coefficient):
    """
    Derive the output path for a full-length single-coefficient steering run.

    Args:
        model_key: Config key for the model.
        steering_layer: Layer index used for steering.
        steering_coefficient: Steering coefficient value.

    Returns:
        Path string such as negsteer_layer14_coef2.5_refusal.json.
    """
    return build_ablate_probe_output_path(
        model_key,
        DIRECTION_SOURCE_REFUSAL,
        ablation_method=ABLATION_METHOD_STEER,
        is_coefficient_sweep=False,
        steering_layer=steering_layer,
        steering_coefficient=steering_coefficient,
    )


def select_top_steering_coefficients(sweep_record, top_k):
    """
    Rank steering coefficients from a sweep by recovery count and mean ROUGE-L.

    Args:
        sweep_record: Parsed sweep JSON with summary.per_coefficient entries.
        top_k: Number of top coefficients to return.

    Returns:
        List of steering coefficient floats, best first.
    """
    per_coefficient = sweep_record["summary"]["per_coefficient"]
    ranked_entries = sorted(
        per_coefficient,
        key=lambda entry: (entry["count_above_0.3"], entry["mean_rouge_l"]),
        reverse=True,
    )
    return [
        entry["steering_coefficient"] for entry in ranked_entries[:top_k]
    ]


def load_top_steering_coefficients_from_sweep(sweep_output_path, top_k):
    """
    Read a sweep JSON file and return its top-ranked steering coefficients.

    Args:
        sweep_output_path: Path to a coefficient sweep results JSON file.
        top_k: Number of top coefficients to return.

    Returns:
        List of steering coefficient floats, best first.
    """
    sweep_record = json.loads(Path(sweep_output_path).read_text(encoding="utf-8"))
    return select_top_steering_coefficients(sweep_record, top_k)


def build_batch_summary(model_entries):
    """
    Build a batch summary record from per-model screening and confirmation results.

    Args:
        model_entries: List of per-model result dicts accumulated during the batch.

    Returns:
        Summary dict with timestamp and models list.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": model_entries,
    }


def write_batch_summary(summary_record, summary_output_path):
    """
    Write the batch summary JSON to disk.

    Args:
        summary_record: Summary dict from build_batch_summary().
        summary_output_path: Destination JSON path.

    Returns:
        Path string for the written summary file.
    """
    summary_output_path = Path(summary_output_path)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(json.dumps(summary_record, indent=2) + "\n")
    print(f"Saved NPO refusal ablation summary to {summary_output_path}", flush=True)
    return str(summary_output_path)


def run_refusal_direction_step(
    model_key,
    direction_num_questions,
    steering_layer,
    skip_existing,
    dry_run,
):
    """
    Extract and save refusal directions for one model if needed.

    Args:
        model_key: Config key from config/models.yaml.
        direction_num_questions: Questions per split for direction extraction.
        steering_layer: Layer index for the auto-generated projection plot.
        skip_existing: Skip when the directions .pt file already exists.
        dry_run: Print planned work without loading the model.

    Returns:
        Dict with status, paths, and optional error message.
    """
    model_entry = get_model(model_key)
    directions_output_path = model_entry["outputs"]["refusal_direction"]
    projection_plot_output_path = default_refusal_direction_projection_plot_path(
        model_entry, model_key, steering_layer
    )
    step_result = {
        "step": "refusal_direction",
        "status": "pending",
        "directions_output_path": directions_output_path,
        "projection_plot_output_path": projection_plot_output_path,
    }

    if skip_existing and Path(directions_output_path).exists():
        step_result["status"] = "skipped_existing"
        print(f"Skipping existing refusal direction for {model_key}", flush=True)
        return step_result

    if dry_run:
        step_result["status"] = "dry_run"
        print(
            f"Would extract refusal direction for {model_key} -> {directions_output_path}",
            flush=True,
        )
        return step_result

    try:
        extract_refusal_directions(
            model_id=model_entry["hf_id"],
            num_questions=direction_num_questions,
            output_path=directions_output_path,
            model_key=model_key,
            projection_plot_layer=steering_layer,
            projection_plot_output_path=projection_plot_output_path,
        )
        step_result["status"] = "completed"
    except Exception as error:
        step_result["status"] = "failed"
        step_result["error"] = str(error)
        print(f"Failed refusal direction for {model_key}: {error}", flush=True)

    return step_result


def run_screen_sweep_step(
    model_key,
    directions_file,
    probe_file,
    screen_num_questions,
    max_new_tokens,
    steering_layer,
    steering_coefficients,
    repetition_penalty,
    skip_existing,
    dry_run,
):
    """
    Run a low-question steering coefficient sweep for one model.

    Args:
        model_key: Config key from config/models.yaml.
        directions_file: Path to saved refusal directions.
        probe_file: Path to unsteered probe JSON for baseline answers.
        screen_num_questions: Number of forget-set questions to probe per coefficient.
        max_new_tokens: Maximum generated tokens per question.
        steering_layer: Layer index for steering.
        steering_coefficients: Coefficient values to sweep.
        repetition_penalty: Generation repetition penalty.
        skip_existing: Skip when the sweep JSON already exists.
        dry_run: Print planned work without loading the model.

    Returns:
        Dict with status, paths, selected metadata, and optional error message.
    """
    model_entry = get_model(model_key)
    sweep_output_path = default_screen_sweep_output_path(
        model_key, steering_layer, screen_num_questions
    )
    step_result = {
        "step": "screen_sweep",
        "status": "pending",
        "sweep_output_path": sweep_output_path,
        "screen_num_questions": screen_num_questions,
        "steering_coefficients": steering_coefficients,
    }

    if skip_existing and Path(sweep_output_path).exists():
        sweep_record = json.loads(Path(sweep_output_path).read_text(encoding="utf-8"))
        step_result["status"] = "skipped_existing"
        step_result["summary"] = sweep_record["summary"]
        print(f"Skipping existing screen sweep for {model_key}", flush=True)
        return step_result

    if dry_run:
        step_result["status"] = "dry_run"
        print(
            f"Would run screen sweep for {model_key} -> {sweep_output_path}",
            flush=True,
        )
        return step_result

    try:
        sweep_record = run_coefficient_sweep(
            model_id=model_entry["hf_id"],
            directions_file=directions_file,
            num_questions=screen_num_questions,
            max_new_tokens=max_new_tokens,
            directions_source=DIRECTION_SOURCE_REFUSAL,
            steering_layer=steering_layer,
            steering_coefficients=steering_coefficients,
            repetition_penalty=repetition_penalty,
            probe_file=probe_file,
            output_path=sweep_output_path,
            model_key=model_key,
        )
        step_result["status"] = "completed"
        step_result["summary"] = sweep_record["summary"]
    except Exception as error:
        step_result["status"] = "failed"
        step_result["error"] = str(error)
        print(f"Failed screen sweep for {model_key}: {error}", flush=True)

    return step_result


def run_confirm_steering_step(
    model_key,
    directions_file,
    probe_file,
    confirm_num_questions,
    max_new_tokens,
    steering_layer,
    steering_coefficient,
    repetition_penalty,
    skip_existing,
    dry_run,
):
    """
    Run a full-length single-coefficient steering probe for one model.

    Args:
        model_key: Config key from config/models.yaml.
        directions_file: Path to saved refusal directions.
        probe_file: Path to unsteered probe JSON for baseline answers.
        confirm_num_questions: Number of forget-set questions to probe.
        max_new_tokens: Maximum generated tokens per question.
        steering_layer: Layer index for steering.
        steering_coefficient: Steering coefficient to apply.
        repetition_penalty: Generation repetition penalty.
        skip_existing: Skip when the confirm JSON already exists.
        dry_run: Print planned work without loading the model.

    Returns:
        Dict with status, paths, summary metrics, and optional error message.
    """
    model_entry = get_model(model_key)
    confirm_output_path = default_confirm_steering_output_path(
        model_key, steering_layer, steering_coefficient
    )
    step_result = {
        "step": "confirm_steering",
        "status": "pending",
        "steering_coefficient": steering_coefficient,
        "confirm_output_path": confirm_output_path,
        "confirm_num_questions": confirm_num_questions,
    }

    if skip_existing and Path(confirm_output_path).exists():
        confirm_record = json.loads(Path(confirm_output_path).read_text(encoding="utf-8"))
        step_result["status"] = "skipped_existing"
        step_result["summary"] = confirm_record["summary"]
        print(
            f"Skipping existing confirm run for {model_key} coef={steering_coefficient}",
            flush=True,
        )
        return step_result

    if dry_run:
        step_result["status"] = "dry_run"
        print(
            f"Would confirm {model_key} coef={steering_coefficient} -> {confirm_output_path}",
            flush=True,
        )
        return step_result

    try:
        confirm_record = ablate_and_probe(
            model_id=model_entry["hf_id"],
            directions_file=directions_file,
            num_questions=confirm_num_questions,
            max_new_tokens=max_new_tokens,
            ablation_method=ABLATION_METHOD_STEER,
            directions_source=DIRECTION_SOURCE_REFUSAL,
            steering_layer=steering_layer,
            steering_coefficient=steering_coefficient,
            repetition_penalty=repetition_penalty,
            probe_file=probe_file,
            output_path=confirm_output_path,
            model_key=model_key,
        )
        step_result["status"] = "completed"
        step_result["summary"] = confirm_record["summary"]
    except Exception as error:
        step_result["status"] = "failed"
        step_result["error"] = str(error)
        print(
            f"Failed confirm run for {model_key} coef={steering_coefficient}: {error}",
            flush=True,
        )

    return step_result


def run_batch_npo_refusal_ablate(
    model_keys,
    direction_num_questions,
    screen_num_questions,
    confirm_num_questions,
    max_new_tokens,
    steering_layer,
    steering_coefficients,
    repetition_penalty,
    top_k,
    skip_existing,
    dry_run,
    run_screen,
    run_confirm,
    summary_output_path,
):
    """
    Run refusal-direction extraction, screening sweeps, and full confirm runs.

    Args:
        model_keys: List of config model keys to process.
        direction_num_questions: Questions per split for refusal direction extraction.
        screen_num_questions: Questions per coefficient during the screening sweep.
        confirm_num_questions: Questions for full confirm runs on top coefficients.
        max_new_tokens: Maximum generated tokens per question.
        steering_layer: Layer index for steering and projection plot.
        steering_coefficients: Coefficients to sweep during screening.
        repetition_penalty: Generation repetition penalty.
        top_k: Number of top coefficients to confirm at full question count.
        skip_existing: Skip steps whose output files already exist.
        dry_run: Print planned work without loading models.
        run_screen: Whether to run direction extraction and screening sweeps.
        run_confirm: Whether to run full confirm runs for top coefficients.
        summary_output_path: Path to write the batch summary JSON.

    Returns:
        Summary dict with per-model step results.
    """
    model_entries = []

    for index, model_key in enumerate(model_keys, start=1):
        print(f"[{index}/{len(model_keys)}] Processing {model_key}", flush=True)
        model_entry = get_model(model_key)
        directions_file = model_entry["outputs"]["refusal_direction"]
        probe_file = model_entry["outputs"]["probe"]
        model_result = {
            "model_key": model_key,
            "hf_id": model_entry["hf_id"],
            "probe_file": probe_file,
            "directions_file": directions_file,
            "screen": None,
            "confirm": [],
        }

        if run_screen:
            direction_step = run_refusal_direction_step(
                model_key=model_key,
                direction_num_questions=direction_num_questions,
                steering_layer=steering_layer,
                skip_existing=skip_existing,
                dry_run=dry_run,
            )
            screen_step = None
            if direction_step["status"] != "failed":
                if not dry_run and not Path(directions_file).exists():
                    direction_step["status"] = "failed"
                    direction_step["error"] = (
                        f"Directions file not found after extraction: {directions_file}"
                    )
                else:
                    screen_step = run_screen_sweep_step(
                        model_key=model_key,
                        directions_file=directions_file,
                        probe_file=probe_file,
                        screen_num_questions=screen_num_questions,
                        max_new_tokens=max_new_tokens,
                        steering_layer=steering_layer,
                        steering_coefficients=steering_coefficients,
                        repetition_penalty=repetition_penalty,
                        skip_existing=skip_existing,
                        dry_run=dry_run,
                    )
            model_result["screen"] = {
                "refusal_direction": direction_step,
                "coefficient_sweep": screen_step,
            }

        sweep_output_path = default_screen_sweep_output_path(
            model_key, steering_layer, screen_num_questions
        )
        if run_confirm:
            if dry_run and run_screen:
                model_result["confirm"] = [
                    {
                        "step": "confirm_steering",
                        "status": "dry_run_deferred",
                        "message": (
                            "Confirm runs will execute after screening selects top "
                            f"{top_k} coefficients."
                        ),
                    }
                ]
            else:
                selected_coefficients = []
                if Path(sweep_output_path).exists():
                    selected_coefficients = load_top_steering_coefficients_from_sweep(
                        sweep_output_path, top_k
                    )
                elif run_screen and model_result.get("screen", {}).get(
                    "coefficient_sweep"
                ):
                    sweep_step = model_result["screen"]["coefficient_sweep"]
                    if sweep_step and sweep_step.get("summary") is not None:
                        selected_coefficients = select_top_steering_coefficients(
                            {"summary": sweep_step["summary"]},
                            top_k,
                        )

                if not selected_coefficients:
                    model_result["confirm_error"] = (
                        f"No screen sweep results found at {sweep_output_path}"
                    )
                    print(model_result["confirm_error"], flush=True)
                else:
                    model_result["selected_coefficients"] = selected_coefficients
                    print(
                        f"Selected top {top_k} coefficients for {model_key}: "
                        f"{selected_coefficients}",
                        flush=True,
                    )
                    for steering_coefficient in selected_coefficients:
                        confirm_step = run_confirm_steering_step(
                            model_key=model_key,
                            directions_file=directions_file,
                            probe_file=probe_file,
                            confirm_num_questions=confirm_num_questions,
                            max_new_tokens=max_new_tokens,
                            steering_layer=steering_layer,
                            steering_coefficient=steering_coefficient,
                            repetition_penalty=repetition_penalty,
                            skip_existing=skip_existing,
                            dry_run=dry_run,
                        )
                        model_result["confirm"].append(confirm_step)

        model_entries.append(model_result)

    summary_record = build_batch_summary(model_entries)
    if not dry_run:
        write_batch_summary(summary_record, summary_output_path)
    return summary_record
