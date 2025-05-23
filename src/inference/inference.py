from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import argparse
from threading import Thread


def tokenize_prompt(prompt, tokenizer):
    """
    Tokenizes the input prompt using the provided tokenizer.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(args.device)
    return input_ids


def main(args):
    print("Device:", args.device)

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)

    # Tokenize the input prompt
    input_ids = tokenize_prompt(args.prompt, tokenizer)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=True,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the generated tokens
    print("-" * 40)
    print(args.model_name)
    print("-" * 40)
    for token in streamer:
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for LeCarnet models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="MaxLSB/LeCarnet-2M",
        help="Name of the model to use for inference",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Il était une fois",
        help="Prompt to use for text generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Length of the generated text",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda or cpu)",
    )
    args = parser.parse_args()
    main(args)
