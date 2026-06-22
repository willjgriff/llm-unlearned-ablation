#!/usr/bin/env python3
"""
ablate_and_probe.py - ablate a saved refusal direction and probe forget-set questions.

Loads per-layer directions from refusal_direction.py, applies directional ablation
at every layer (via forward hooks or weight orthogonalisation), then runs forget-set
questions and saves model answers alongside ground-truth to JSON.

Setup:
    pip install -r requirements.txt

Usage:
    python src/ablate_and_probe.py --model-key npo_unlearned
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, RepetitionPenaltyLogitsProcessor

from ablation import (
    apply_weight_orthogonalisation,
    register_ablation_hooks_on_all_layers,
    register_steering_hook,
    remove_ablation_hooks,
    restore_original_weights,
)
from model_config import get_model

FORGET_SPLIT = "forget10"
ABLATION_METHOD_HOOKS = "hooks"
ABLATION_METHOD_ORTHOGONALISATION = "orthogonalisation"
ABLATION_METHOD_STEER = "steer"
DIRECTION_SOURCE_REFUSAL = "refusal"
DIRECTION_SOURCE_CONFABULATION = "confabulation"
DIRECTION_SOURCE_CONFIG_KEYS = {
    DIRECTION_SOURCE_REFUSAL: "refusal_direction",
    DIRECTION_SOURCE_CONFABULATION: "confabulation_direction",
}


class MpsSafeRepetitionPenaltyLogitsProcessor:
    """
    Apply repetition penalty on CPU to avoid an MPS scatter/gather bug.

    Transformers' RepetitionPenaltyLogitsProcessor can trigger an MPS assertion
    (NDArray > 2**32 bytes) when scores live on MPS. Running the penalty on CPU
    and copying back preserves behaviour without the crash.

    Args:
        penalty: Repetition penalty value (1.0 means no penalty).
    """

    def __init__(self, penalty):
        self.repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
            penalty=penalty
        )

    def __call__(self, input_ids, scores):
        """
        Apply repetition penalty to logits, using CPU when scores are on MPS.

        Args:
            input_ids: Generated token ids so far.
            scores: Next-token logits tensor.

        Returns:
            Processed logits on the same device and dtype as the input scores.
        """
        scores_device = scores.device
        scores_dtype = scores.dtype
        processed_scores = self.repetition_penalty_processor(
            input_ids.cpu(), scores.float().cpu()
        )
        return processed_scores.to(device=scores_device, dtype=scores_dtype)


def build_generate_kwargs(
    tokenizer, max_new_tokens, repetition_penalty, device
):
    """
    Build kwargs for model.generate, using an MPS-safe repetition penalty path.

    Args:
        tokenizer: Tokenizer used for pad_token_id.
        max_new_tokens: Maximum tokens to generate.
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty).
        device: Torch device string for the model.

    Returns:
        Dict of keyword arguments for model.generate.
    """
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if repetition_penalty == 1.0:
        return generate_kwargs

    if device == "mps":
        generate_kwargs["logits_processor"] = [
            MpsSafeRepetitionPenaltyLogitsProcessor(repetition_penalty)
        ]
    else:
        generate_kwargs["repetition_penalty"] = repetition_penalty

    return generate_kwargs


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


def generate_answer(
    model, tokenizer, question, device, max_new_tokens, repetition_penalty=1.0
):
    """
    Run greedy generation for a single TOFU question using the chat template.

    Args:
        model: Loaded causal LM in eval mode (may have ablation hooks registered).
        tokenizer: Matching tokenizer with chat template.
        question: User question text from the TOFU dataset.
        device: Torch device string (cuda, mps, or cpu).
        max_new_tokens: Maximum tokens to generate.
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty).

    Returns:
        Decoded model answer string with special tokens stripped.
    """
    messages = [{"role": "user", "content": question}]
    input_token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    generate_kwargs = build_generate_kwargs(
        tokenizer, max_new_tokens, repetition_penalty, device
    )

    with torch.no_grad():
        output_token_ids = model.generate(input_token_ids, **generate_kwargs)

    generated_token_ids = output_token_ids[0, input_token_ids.shape[1]:]
    return tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()


def load_all_direction_vectors(directions_file):
    """
    Load all per-layer direction vectors from a saved directions file.

    Args:
        directions_file: Path to a .pt file from refusal_direction.py or
            confabulation_direction.py.

    Returns:
        List of direction tensors, one per layer.
    """
    saved_record = torch.load(directions_file, map_location="cpu", weights_only=False)
    return saved_record["directions"]


def print_direction_norms(direction_vectors, layer_indices):
    """
    Print the L2 norm of direction vectors at selected layer indices.

    Args:
        direction_vectors: List of direction tensors, one per layer.
        layer_indices: Layer indices to sample and print norms for.
    """
    for layer_index in layer_indices:
        direction_norm = direction_vectors[layer_index].norm().item()
        print(f"Layer {layer_index} direction norm: {direction_norm:.4f}", flush=True)


def resolve_directions_file(model_entry, directions_file_argument, directions_source):
    """
    Resolve the directions file path from CLI args and model config.

    Args:
        model_entry: Model dict from config/models.yaml.
        directions_file_argument: Explicit --directions-file path, if any.
        directions_source: Either 'refusal' or 'confabulation'.

    Returns:
        Path string to the directions .pt file.
    """
    if directions_file_argument:
        return directions_file_argument
    if model_entry.get("directions_file"):
        return model_entry["directions_file"]

    config_output_key = DIRECTION_SOURCE_CONFIG_KEYS[directions_source]
    model_outputs = model_entry["outputs"]
    if config_output_key not in model_outputs:
        raise KeyError(
            f"Model '{model_entry.get('hf_id', 'unknown')}' has no "
            f"outputs.{config_output_key} in config/models.yaml."
        )
    return model_outputs[config_output_key]


def resolve_probe_file(model_entry, probe_file_argument):
    """
    Resolve the probe results JSON path from CLI args and model config.

    Args:
        model_entry: Model dict from config/models.yaml.
        probe_file_argument: Explicit --probe-file path, if any.

    Returns:
        Path string to the probe JSON file, or None if not configured.
    """
    if probe_file_argument:
        return probe_file_argument
    return model_entry.get("outputs", {}).get("probe")


def load_probe_answers_by_index(probe_file):
    """
    Load non-ablated model answers from a tofu_probe.py results file.

    Args:
        probe_file: Path to a probe JSON file, or None.

    Returns:
        Tuple of (index-to-answer dict, probe file path used). The dict is empty
        when no probe file is available.
    """
    if probe_file is None:
        return {}, None

    probe_file_path = Path(probe_file)
    if not probe_file_path.is_file():
        print(
            f"Warning: probe file not found at {probe_file_path}; "
            f"probe_answer will be omitted.",
            flush=True,
        )
        return {}, str(probe_file_path)

    probe_record = json.loads(probe_file_path.read_text(encoding="utf-8"))
    probe_answers_by_index = {
        entry["index"]: entry["model_answer"] for entry in probe_record["results"]
    }
    print(
        f"Loaded {len(probe_answers_by_index)} non-ablated answers from {probe_file_path}.",
        flush=True,
    )
    return probe_answers_by_index, str(probe_file_path)


def default_output_path(model_entry, directions_source):
    """
    Derive the default ablate-and-probe output path for a directions source.

    Confabulation runs use a separate _confab suffix so they do not overwrite
    refusal-direction ablation results.

    Args:
        model_entry: Model dict from config/models.yaml.
        directions_source: Either 'refusal' or 'confabulation'.

    Returns:
        Default output path string, or None if the model has no ablate_and_probe output.
    """
    if "ablate_and_probe" not in model_entry["outputs"]:
        return None

    base_output_path = Path(model_entry["outputs"]["ablate_and_probe"])
    if directions_source == DIRECTION_SOURCE_CONFABULATION:
        return str(
            base_output_path.with_name(
                base_output_path.stem + "_confab" + base_output_path.suffix
            )
        )
    return str(base_output_path)


def ablate_and_probe(
    model_id,
    directions_file,
    num_questions,
    max_new_tokens,
    ablation_method=ABLATION_METHOD_HOOKS,
    directions_source=DIRECTION_SOURCE_REFUSAL,
    steering_layer=None,
    steering_coefficient=1.0,
    repetition_penalty=1.0,
    probe_file=None,
    output_path=None,
    model_key=None,
):
    """
    Ablate the refusal direction at every layer and probe forget-set questions.

    Args:
        model_id: Hugging Face model id or local path.
        directions_file: Path to saved per-layer direction vectors.
        num_questions: Number of questions to probe from the start of the forget10 split.
        max_new_tokens: Maximum tokens to generate per question.
        ablation_method: Either 'hooks' (forward-hook ablation), 'orthogonalisation'
            (in-place weight orthogonalisation), or 'steer' (single-layer negative
            steering via activation subtraction).
        directions_source: Which direction type was loaded ('refusal' or 'confabulation').
        steering_layer: Layer index for single-layer steering (required when method is steer).
        steering_coefficient: Scalar multiplier for the steering direction (steer only).
        repetition_penalty: Penalty for repeating tokens during generation (1.0 = no penalty).
        probe_file: Optional path to a tofu_probe.py JSON file for non-ablated answers.
        output_path: Optional path to write structured JSON results.
        model_key: Optional config key from config/models.yaml.

    Returns:
        Dict containing run metadata and per-question results.
    """
    device, model_dtype = resolve_device_and_dtype()

    print(f"Loading model on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=model_dtype
    ).to(device)
    model.eval()
    print("Model loaded.", flush=True)

    direction_vectors = load_all_direction_vectors(directions_file)
    num_layers_ablated = len(model.model.layers)
    if len(direction_vectors) != num_layers_ablated:
        raise ValueError(
            f"Directions file has {len(direction_vectors)} layers but model has "
            f"{num_layers_ablated}."
        )

    ablation_handles = None
    saved_weights = None
    if ablation_method == ABLATION_METHOD_HOOKS:
        ablation_handles = register_ablation_hooks_on_all_layers(
            model, direction_vectors, device, model_dtype
        )
        print(
            f"Ablating {directions_source} direction at all {num_layers_ablated} layers "
            f"via {ablation_method}.",
            flush=True,
        )
    elif ablation_method == ABLATION_METHOD_ORTHOGONALISATION:
        saved_weights = apply_weight_orthogonalisation(
            model, direction_vectors, device, model_dtype
        )
        print(
            f"Ablating {directions_source} direction at all {num_layers_ablated} layers "
            f"via {ablation_method}.",
            flush=True,
        )
    elif ablation_method == ABLATION_METHOD_STEER:
        if steering_layer is None:
            raise ValueError(
                "steering_layer is required when ablation_method is 'steer'."
            )
        ablation_handles = register_steering_hook(
            model,
            direction_vectors,
            steering_layer,
            steering_coefficient,
            device,
            model_dtype,
        )
        print(
            f"Steering {directions_source} direction at layer {steering_layer} "
            f"(coefficient {steering_coefficient}) via {ablation_method}.",
            flush=True,
        )
    else:
        raise ValueError(
            f"Unknown ablation method '{ablation_method}'; expected "
            f"'{ABLATION_METHOD_HOOKS}', '{ABLATION_METHOD_ORTHOGONALISATION}', "
            f"or '{ABLATION_METHOD_STEER}'."
        )

    sampled_layer_indices = list(range(0, num_layers_ablated, 4))
    print_direction_norms(direction_vectors, sampled_layer_indices)

    probe_answers_by_index, probe_file_used = load_probe_answers_by_index(probe_file)

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
    try:
        for index in progress:
            question = dataset[index]["question"]
            ground_truth_answer = dataset[index]["answer"]

            progress.set_postfix_str(f"asking {index + 1}/{question_count}", refresh=True)
            model_answer = generate_answer(
                model,
                tokenizer,
                question,
                device,
                max_new_tokens,
                repetition_penalty=repetition_penalty,
            )
            progress.set_postfix_str(f"answered {index + 1}/{question_count}", refresh=True)

            result_entry = {
                "index": index,
                "question": question,
                "ground_truth": ground_truth_answer,
                "model_answer": model_answer,
            }
            if index in probe_answers_by_index:
                result_entry["probe_answer"] = probe_answers_by_index[index]
            results.append(result_entry)
    finally:
        if ablation_handles is not None:
            remove_ablation_hooks(ablation_handles)
        if saved_weights is not None:
            restore_original_weights(model, saved_weights)

    run_record = {
        "model": model_id,
        "model_key": model_key,
        "directions_file": directions_file,
        "directions_source": directions_source,
        "ablation_method": ablation_method,
        "steering_layer": steering_layer,
        "steering_coefficient": steering_coefficient,
        "repetition_penalty": repetition_penalty,
        "probe_file": probe_file_used,
        "num_layers_ablated": num_layers_ablated,
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
    """Parse CLI arguments and run ablation probing."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        required=True,
        help="short name from config/models.yaml (e.g. npo_unlearned)",
    )
    parser.add_argument(
        "--directions-file",
        default=None,
        help="path to directions .pt file (overrides --directions-source)",
    )
    parser.add_argument(
        "--directions-source",
        choices=[DIRECTION_SOURCE_REFUSAL, DIRECTION_SOURCE_CONFABULATION],
        default=DIRECTION_SOURCE_REFUSAL,
        help="which saved direction file to load from config (default: refusal)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ablation-method",
        choices=[
            ABLATION_METHOD_HOOKS,
            ABLATION_METHOD_ORTHOGONALISATION,
            ABLATION_METHOD_STEER,
        ],
        default=ABLATION_METHOD_HOOKS,
        help="how to apply directional intervention (default: hooks)",
    )
    parser.add_argument(
        "--steering-coefficient",
        type=float,
        default=1.0,
        help="multiplier for the steering direction when --ablation-method steer (default: 1.0)",
    )
    parser.add_argument(
        "--steering-layer",
        type=int,
        default=None,
        help="layer index for single-layer steering (required when --ablation-method steer)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="number of questions to probe",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="penalty for repeating tokens during generation (default: 1.0)",
    )
    parser.add_argument(
        "--probe-file",
        default=None,
        help="path to probe JSON for non-ablated answers (default from config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to write JSON results (default from config; pass empty string to skip)",
    )
    arguments = parser.parse_args()

    if arguments.layer is not None:
        print(
            "Warning: --layer is ignored; use --steering-layer with --ablation-method steer.",
            flush=True,
        )

    if arguments.ablation_method == ABLATION_METHOD_STEER:
        if arguments.steering_layer is None:
            parser.error("--steering-layer is required when --ablation-method is steer")

    model_entry = get_model(arguments.model_key)
    directions_file = resolve_directions_file(
        model_entry,
        arguments.directions_file,
        arguments.directions_source,
    )
    output_path = arguments.output
    if output_path is None:
        output_path = default_output_path(model_entry, arguments.directions_source)
    elif output_path == "":
        output_path = None

    probe_file = resolve_probe_file(model_entry, arguments.probe_file)

    ablate_and_probe(
        model_id=model_entry["hf_id"],
        directions_file=directions_file,
        num_questions=arguments.num_questions,
        max_new_tokens=arguments.max_new_tokens,
        ablation_method=arguments.ablation_method,
        directions_source=arguments.directions_source,
        steering_layer=arguments.steering_layer,
        steering_coefficient=arguments.steering_coefficient,
        repetition_penalty=arguments.repetition_penalty,
        probe_file=probe_file,
        output_path=output_path,
        model_key=arguments.model_key,
    )


if __name__ == "__main__":
    main()
