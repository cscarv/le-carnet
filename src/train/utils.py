import torch


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
