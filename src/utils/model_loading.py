"""Model and tokenizer loading helpers."""

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.device import resolve_device_and_dtype


def load_model_and_tokenizer(model_id):
    """
    Load a causal LM and matching tokenizer on the best available device.

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
