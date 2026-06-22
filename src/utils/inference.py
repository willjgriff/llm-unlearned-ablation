"""Model loading and greedy generation helpers for TOFU probing."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def load_model_and_tokenizer(model_id):
    """
    Load a causal LM and matching tokenizer in eval mode on the best available device.

    Args:
        model_id: Hugging Face model id or local path.

    Returns:
        Tuple of (model, tokenizer, device, model_dtype).
    """
    device, model_dtype = resolve_device_and_dtype()

    print(f"Loading model on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=model_dtype
    ).to(device)
    model.eval()
    print("Model loaded.", flush=True)

    return model, tokenizer, device, model_dtype


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
