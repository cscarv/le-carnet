import os
import asyncio
import time
from openai import AsyncOpenAI, RateLimitError
import random
import jsonlines
from tqdm.asyncio import tqdm_asyncio
import argparse
from tqdm import tqdm


class Vocabulary:
    def __init__(self, vocab_dir="vocabulary"):
        categories = ["adjectives", "nouns", "verbs", "features"]
        for category in categories:
            file_path = os.path.join(vocab_dir, f"{category}.txt")
            with open(file_path, "r") as f:
                setattr(self, category, [line.strip() for line in f if line.strip()])

    def random_choice(self, category):
        return random.choice(getattr(self, category))

    def random_features(self, count=2):
        return random.sample(self.features, count)


def build_batch(vocab: Vocabulary, batch_size: int) -> list[list[dict]]:
    """
    Build a batch of message lists, each with a unique prompt using random words.
    """
    batch = []
    for _ in range(batch_size):
        verb = vocab.random_choice("verbs")
        noun = vocab.random_choice("nouns")
        adjective = vocab.random_choice("adjectives")
        feature_1, feature_2 = vocab.random_features(2)

        prompt = f"""
        Écris une courte histoire (3 à 5 paragraphes) qui n’utilise que des mots très simples qu’un enfant de 3 ans comprendrait. 
        L’histoire doit s’articuler autour du verbe « {verb} », du nom « {noun} » et de l’adjectif « {adjective} ». 
        L'histoire doit avoir les caractéristiques suivantes : {feature_1}, {feature_2}. 
        Souviens-toi de n’utiliser que des mots simples !"
        """

        batch.append([{"role": "user", "content": prompt}])

    return batch


def get_async_client(api_key: str, base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


async def send_message(client, messages, model_name):
    backoff_time = 1
    for _ in range(3):
        try:
            response = await client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.7,
                max_tokens=512,
                top_p=0.95,
            )
            return response.choices[0].message.content

        except RateLimitError:
            await asyncio.sleep(backoff_time)
            backoff_time *= 2
    raise RuntimeError("Max retries reached. Halting execution.")


async def save_stories_to_jsonl(stories: list[str], file_path: str):
    """
    Appends a list of stories to a JSONL file, with the format {"text": Story}.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode="a", encoding="utf-8") as f:
        writer = jsonlines.Writer(f)
        writer.write_all({"text": story} for story in stories)


async def generate_batches(
    client: AsyncOpenAI,
    vocab: Vocabulary,
    model_name: str,
    total_requests: int,
    batch_size: int,
    concurrency: int,
    output_file: str,
) -> list[dict]:
    """
    Generates stories in parallel by chunking the total requests into manageable async batches.
    """

    semaphore = asyncio.Semaphore(concurrency)
    num_batches = total_requests // batch_size

    async def run_single_batch():
        async with semaphore:
            messages_batch = build_batch(vocab, batch_size)
            return await asyncio.gather(
                *[
                    send_message(
                        client,
                        msgs,
                        model_name,
                    )
                    for msgs in messages_batch
                ]
            )

    batches = [run_single_batch() for _ in range(num_batches)]
    start = time.time()
    stories_generated = 0

    # Loop through the batches and await their completion
    for batch_res in tqdm_asyncio.as_completed(
        batches, total=num_batches, desc="Generating"
    ):
        batch_output = await batch_res
        await save_stories_to_jsonl(batch_output, output_file)
        stories_generated += len(batch_output)

    elapsed = time.time() - start
    tqdm.write(f"> {stories_generated} stories generated in {elapsed:.2f} seconds.")


def generate_output_file(output_dir, model_name: str) -> str:
    """
    Generates a unique output file name based on the model name and current timestamp.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(output_dir, f"stories_{model_name}_{timestamp}.jsonl")


async def main(args):
    """
    Main function to set up the client and generate stories.
    """

    # Set the API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"API key not found. Set the OPENAI_API_KEY environment variable."
        )

    # Initialize the vocabulary and client
    vocab = Vocabulary()
    client = get_async_client(api_key=api_key, base_url=args.base_url)

    output_file = generate_output_file(args.output_dir, args.model_name)

    # Generate batches of stories
    await generate_batches(
        client,
        vocab,
        args.model_name,
        args.total_requests,
        args.batch_size,
        args.concurrency,
        output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stories with a LLM.")
    parser.add_argument("--base_url", type=str, required=True, help="Base URL.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to use for generating stories.",
    )
    parser.add_argument(
        "--total_requests",
        type=int,
        default=512,
        help="Total number of requests to make.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for each request."
    )
    parser.add_argument(
        "--concurrency", type=int, default=2, help="Number of concurrent requests."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/generated/openai",
        help="Output file path for storing stories.",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
