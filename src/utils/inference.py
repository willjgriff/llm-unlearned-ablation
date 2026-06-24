"""Greedy generation helpers for TOFU question probing."""

import torch
from transformers import RepetitionPenaltyLogitsProcessor


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


def build_generate_kwargs(tokenizer, max_new_tokens, repetition_penalty, device):
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
