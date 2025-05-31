import torch
from contextlib import nullcontext


def generate_text(
    model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=50
):
    """
    Generate text using the model and tokenizer based on a given prompt.
    """
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
        )

    model.train()
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def num_parameters(model):
    """
    Count the number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_mixed_precision_dtype():
    """
    Determine the appropriate mixed precision dtype based on the available hardware.
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    else:
        return torch.float32


def get_amp_scaler_and_autocast(device: str, dtype: torch.dtype):
    """
    Get the appropriate scaler and autocast context for mixed precision training.
    """
    device = torch.device(device)
    if device.type == "cuda" and dtype is not None:
        autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=dtype)
        scaler = torch.amp.GradScaler() if dtype == torch.float16 else None
    else:
        autocast_ctx = nullcontext()
        scaler = None
    return scaler, autocast_ctx
