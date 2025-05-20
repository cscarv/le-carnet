import torch


def generate_next_tokens(model, idx, context_size, max_new_tokens, k, temperature):
    """
    Generate the next tokens for a given input sequence.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            output = model(idx_cond)
            logits = output.logits

        logits = logits[:, -1, :]
        next_token = sample_next_token(logits, k, temperature)
        idx = torch.cat((idx, next_token), dim=1)

    return idx


def generate_text(
    model,
    tokenizer,
    context_size,
    start_context,
    max_new_tokens=50,
    k=10,
    temperature=0.7,
):
    """
    Generate text using using top-k sampling from a given start context.
    """
    model.eval()
    device = model.device
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        tokens_ids = generate_next_tokens(
            model,
            encoded,
            context_size,
            max_new_tokens=max_new_tokens,
            k=k,
            temperature=temperature,
        )

    decoded_text = token_ids_to_text(tokens_ids, tokenizer)
    cleaned_text = decoded_text.replace("\n", " ")
    model.train()
    return cleaned_text


def sample_next_token(logits, k=10, temperature=0.8):
    """
    Sample the next token from the model's logits using top-k sampling.
    """
    k = min(k, logits.shape[-1])
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probabilities = torch.softmax(top_k_logits, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    return top_k_indices.gather(-1, next_token)


def text_to_token_ids(text, tokenizer):
    """
    Convert text to token IDs using the tokenizer.
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs back to text using the tokenizer.
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def num_parameters(model):
    """
    Count the number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
