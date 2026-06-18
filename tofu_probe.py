#!/usr/bin/env python3
"""
tofu_probe.py - prompt a TOFU (un)learned checkpoint with forget-set author
questions and print the model's answer next to the TOFU ground-truth answer.

Setup:
    pip install -r requirements.txt

Usage (run once per model to compare behavior):
    # baseline - the model that still knows the fake authors
    python tofu_probe.py --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full

    # output-preference unlearning - your "should recover after ablation" cases
    python tofu_probe.py --model open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget01_NPO_lr2e-05_beta0.5_alpha1_epoch10

Compare the *_full output against an unlearned checkpoint on the SAME questions:
that contrast is the behavioral signal you'll later try to reverse by ablating
the refusal direction.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

DEBUG_LOG_PATH = Path(__file__).resolve().parent / ".cursor" / "debug-82cd0e.log"


def debug_log(hypothesis_id, location, message, data=None, run_id="pre-fix"):
    """Append one NDJSON debug log line for this debug session."""
    # region agent log
    payload = {
        "sessionId": "82cd0e",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload) + "\n")
    # endregion


def inspect_huggingface_model(model_id):
    """
    Check Hugging Face auth state and whether a model repo is reachable.

    Args:
        model_id: Hugging Face model id or local path.

    Returns:
        Dict with auth and repository inspection metadata.
    """
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    cached_token_present = token_file.exists()

    inspection = {
        "model_id": model_id,
        "token_env_present": token_present,
        "token_file_present": cached_token_present,
        "is_local_path": Path(model_id).exists(),
    }

    if inspection["is_local_path"]:
        inspection["repo_status"] = "local_path"
        return inspection

    api = HfApi()
    try:
        model_info = api.model_info(model_id)
        inspection["repo_status"] = "found"
        inspection["repo_private"] = getattr(model_info, "private", None)
        inspection["repo_gated"] = getattr(model_info, "gated", None)
    except Exception as error:
        inspection["repo_status"] = "error"
        inspection["error_type"] = type(error).__name__
        inspection["error_message"] = str(error)

    return inspection


def resolve_device_and_dtype():
    """
    Pick the best available device and matching model dtype for inference.

    Returns:
        Tuple of (device name, torch dtype).
    """
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def generate_answer(model, tokenizer, question, device, max_new_tokens):
    """
    Run greedy generation for a single TOFU question using the chat template.

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Matching tokenizer with chat template.
        question: User question text from the TOFU dataset.
        device: Torch device string (cuda, mps, or cpu).
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Decoded model answer string with special tokens stripped.
    """
    messages = [{"role": "user", "content": question}]
    input_token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_token_ids = model.generate(
            input_token_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_token_ids = output_token_ids[0, input_token_ids.shape[1]:]
    return tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()


def run_probe(model_id, split, num_questions, max_new_tokens, output_path=None):
    """
    Load a checkpoint, probe TOFU forget-set questions, print results, and optionally save JSON.

    Args:
        model_id: Hugging Face model id or local path.
        split: TOFU dataset config name (e.g. forget01).
        num_questions: Number of questions to probe from the start of the split.
        max_new_tokens: Maximum tokens to generate per question.
        output_path: Optional path to write structured JSON results.

    Returns:
        Dict containing run metadata and per-question results.
    """
    device, model_dtype = resolve_device_and_dtype()
    print(f"Loading {model_id} on {device} ...")

    inspection = inspect_huggingface_model(model_id)
    debug_log(
        "A",
        "tofu_probe.py:run_probe:auth",
        "Hugging Face auth state before model load",
        {
            "token_env_present": inspection["token_env_present"],
            "token_file_present": inspection["token_file_present"],
        },
    )
    debug_log(
        "B",
        "tofu_probe.py:run_probe:repo",
        "Hugging Face repo inspection before model load",
        inspection,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=model_dtype
    ).to(device)
    model.eval()

    dataset = load_dataset("locuslab/TOFU", split)["train"]
    question_count = min(num_questions, len(dataset))
    print(
        f"Loaded TOFU '{split}' ({len(dataset)} QA pairs); "
        f"probing first {question_count}.\n"
    )

    results = []
    for index in range(question_count):
        question = dataset[index]["question"]
        ground_truth_answer = dataset[index]["answer"]
        model_answer = generate_answer(
            model, tokenizer, question, device, max_new_tokens
        )

        print(f"[{index + 1}] Q: {question}")
        print(f"    GROUND TRUTH: {ground_truth_answer}")
        print(f"    MODEL:        {model_answer}\n")

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
        "split": split,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": question_count,
        "results": results,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(run_record, indent=2) + "\n")
        print(f"Saved results to {output_path}")

    return run_record


def main():
    """Parse CLI arguments and run the TOFU forget-set probe."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument(
        "--split",
        default="forget01",
        help="TOFU config: forget01 / forget05 / forget10 / retain90 / full",
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
        default="results/baseline_full_forget01.json",
        help="path to write JSON results (pass empty string to skip saving)",
    )
    arguments = parser.parse_args()

    output_path = arguments.output or None
    run_probe(
        model_id=arguments.model,
        split=arguments.split,
        num_questions=arguments.num_questions,
        max_new_tokens=arguments.max_new_tokens,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
