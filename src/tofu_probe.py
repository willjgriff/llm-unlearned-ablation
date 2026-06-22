#!/usr/bin/env python3
"""
tofu_probe.py - prompt a TOFU (un)learned checkpoint with forget-set author
questions and save model answers alongside TOFU ground-truth to JSON.

Setup:
    pip install -r requirements.txt

Usage (run once per model to compare behavior):
    # baseline - the model that still knows the fake authors
    python src/tofu_probe.py --model-key baseline_full

    # output-preference unlearning - your "should recover after ablation" cases
    python src/tofu_probe.py --model-key npo_unlearned

    # one-off checkpoint: add an entry to config/models.yaml first

Compare the *_full output against an unlearned checkpoint on the SAME questions:
that contrast is the behavioral signal you'll later try to reverse by ablating
the refusal direction.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from utils.constants import FORGET_SPLIT
from utils.inference import generate_answer, load_model_and_tokenizer
from utils.model_config import get_model


def run_probe(model_id, num_questions, max_new_tokens, output_path=None, model_key=None):
    """
    Load a checkpoint, probe TOFU forget-set questions, and optionally save JSON.

    Args:
        model_id: Hugging Face model id or local path.
        num_questions: Number of questions to probe from the start of the forget10 split.
        max_new_tokens: Maximum tokens to generate per question.
        output_path: Optional path to write structured JSON results.
        model_key: Optional config key from config/models.yaml.

    Returns:
        Dict containing run metadata and per-question results.
    """
    model, tokenizer, device, _model_dtype = load_model_and_tokenizer(model_id)

    dataset = load_dataset("locuslab/TOFU", FORGET_SPLIT)["train"]
    question_count = min(num_questions, len(dataset))
    print(
        f"Probing {question_count} questions from TOFU '{FORGET_SPLIT}' "
        f"({len(dataset)} available)...",
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
        question = dataset[index]["question"]
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
            }
        )

    run_record = {
        "model": model_id,
        "model_key": model_key,
        "split": FORGET_SPLIT,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": question_count,
        "results": results,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(run_record, indent=2) + "\n")
        print(f"Saved results to {output_path}", flush=True)

    return run_record


def main():
    """Parse CLI arguments and run the TOFU forget-set probe."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        required=True,
        help="short name from config/models.yaml (e.g. baseline_full, npo_unlearned)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="number of questions to probe",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--output",
        default=None,
        help="path to write JSON results (default from config; pass empty string to skip)",
    )
    arguments = parser.parse_args()

    model_entry = get_model(arguments.model_key)
    output_path = arguments.output
    if output_path is None:
        output_path = model_entry["outputs"]["probe"]
    elif output_path == "":
        output_path = None

    run_probe(
        model_id=model_entry["hf_id"],
        num_questions=arguments.num_questions,
        max_new_tokens=arguments.max_new_tokens,
        output_path=output_path,
        model_key=arguments.model_key,
    )


if __name__ == "__main__":
    main()
