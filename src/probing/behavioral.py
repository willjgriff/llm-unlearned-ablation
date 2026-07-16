"""Behavioral TOFU forget-set probing without ablation."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from utils.constants import QUESTION_MODE_ORIGINAL
from utils.inference import generate_answer
from utils.metrics import (
    attach_summary_before_results,
    compute_rouge_l,
    summarize_flat_results,
)
from utils.model_loading import load_model_and_tokenizer
from utils.tofu_data import load_forget_split_dataset, resolve_forget_split_name


def run_probe(
    model_id,
    num_questions,
    max_new_tokens,
    output_path=None,
    model_key=None,
    question_mode=QUESTION_MODE_ORIGINAL,
    split_name=None,
):
    """
    Load a checkpoint, probe TOFU forget-set questions, and optionally save JSON.

    Args:
        model_id: Hugging Face model id or local path.
        num_questions: Number of questions to probe from the start of the forget split.
        max_new_tokens: Maximum tokens to generate per question.
        output_path: Optional path to write structured JSON results.
        model_key: Optional config key from config/models.yaml.
        question_mode: Use original forget questions or paraphrased variants.
        split_name: Optional TOFU config override; inferred from question_mode if omitted.

    Returns:
        Dict containing run metadata and per-question results.
    """
    model, tokenizer, device, _model_dtype = load_model_and_tokenizer(model_id)

    resolved_split = resolve_forget_split_name(question_mode, split_name)
    dataset, question_count, question_field = load_forget_split_dataset(
        num_questions,
        resolved_split,
        question_mode,
    )
    print(
        f"Probing {question_count} {question_mode} questions from TOFU "
        f"'{resolved_split}' ({len(dataset)} available)...",
        flush=True,
    )

    results = []
    progress = tqdm(
        range(question_count),
        desc="Probing",
        unit="question",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    for index in progress:
        question = dataset[index][question_field]
        ground_truth_answer = dataset[index]["answer"]

        progress.set_postfix_str(f"asking {index + 1}/{question_count}", refresh=True)
        model_answer = generate_answer(
            model, tokenizer, question, device, max_new_tokens
        )
        progress.set_postfix_str(f"answered {index + 1}/{question_count}", refresh=True)

        results.append(
            {
                "index": index,
                "question": question,
                "ground_truth": ground_truth_answer,
                "model_answer": model_answer,
                "rouge_l": compute_rouge_l(ground_truth_answer, model_answer),
            }
        )

    run_record = {
        "model": model_id,
        "model_key": model_key,
        "split": resolved_split,
        "question_mode": question_mode,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": question_count,
        "results": results,
    }
    attach_summary_before_results(run_record, summarize_flat_results(results))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(run_record, indent=2) + "\n")
        print(f"Saved results to {output_path}", flush=True)

    return run_record
