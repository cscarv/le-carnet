from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import argparse
from threading import Thread
import time


def tokenize_prompt(prompt, tokenizer, device):
    """
    Tokenizes the input prompt using the provided tokenizer.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    return input_ids


def get_tokenizer(model_name):
    """
    Loads the tokenizer for the specified model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def main(args):
    # Load the model and tokenizer
    tokenizer = get_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)

    # Tokenize the input prompt
    input_ids = tokenize_prompt(args.prompt, tokenizer, args.device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=True,
        streamer=streamer,
        no_repeat_ngram_size=4,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the generated tokens
    print("\n" + "-" * 40)
    print(args.model_name)
    print("-" * 40 + "\n")

    start_time = time.time()
    token_count = 0

    for token in streamer:
        token_ids = tokenizer(token, return_tensors="pt").input_ids[0]
        if tokenizer.eos_token_id in token_ids:
            break
        token_count += 1
        print(token, end="", flush=True)

    print()
    elapsed_time = time.time() - start_time
    print(f"\n> Speed: {token_count / elapsed_time:.2f} tokens/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for LeCarnet models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="MaxLSB/LeCarnet-3M",
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
