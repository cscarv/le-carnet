import os
from tqdm import tqdm
import time
import random
import jsonlines
import argparse
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


class Vocabulary:
    def __init__(self, vocab_dir="src/data/vocabulary"):
        categories = ["adjectives", "nouns", "verbs", "features"]
        for category in categories:
            file_path = os.path.join(vocab_dir, f"{category}.txt")
            with open(file_path, "r") as f:
                setattr(self, category, [line.strip() for line in f if line.strip()])

    def random_choice(self, category):
        return random.choice(getattr(self, category))

    def random_features(self, count=2):
        return random.sample(self.features, count)


def build_message(vocab: Vocabulary):
    """
    Builds a message for the LLM to generate a story.
    """
    verb = vocab.random_choice("verbs")
    noun = vocab.random_choice("nouns")
    adjective = vocab.random_choice("adjectives")
    feature = vocab.random_choice("features")

    prompt = f"""
    Write a short story in French suitable for 5-to-7-year-old children.
    Use simple, easy-to-understand words and limit the story to 3-4 short paragraphs (around 200-300 words).
    The story should feature a clear beginning, middle, and end. Incorporate the verb ”{verb}”, the noun ”{noun}”, and the adjective ”{adjective}” naturally into the story.
    The story should also integrate the conclusion/tone ”{feature}” through actions and outcomes, without directly stating the tone.
    Remember to only use simple words and keep the story short!
    """
    message = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    return message


def get_client(api_key: str):
    return Mistral(api_key=api_key)


def send_message(client, message, model_name):
    """
    Sends a message to the Mistral API with retries and exponential backoff.
    Adds a small random delay to stagger requests.
    """
    time.sleep(random.uniform(0.1, 0.5))  # Stagger requests to avoid rate limit spikes
    backoff = 1.0
    for attempt in range(5):  # 5 attempts total
        try:
            resp = client.chat.complete(
                messages=message,
                model=model_name,
                temperature=0.7,
                max_tokens=512,
                top_p=0.95,
            )
            return resp.choices[0].message.content
        except SDKError as e:
            if getattr(e, "status_code", None) == 429 and attempt < 4:
                wait = backoff + random.random() * 0.5
                time.sleep(wait)
                backoff *= 2
            else:
                raise e
    raise RuntimeError("Max retries reached")


def save_stories_to_jsonl(stories: list[str], file_path: str):
    """
    Appends a list of stories to a JSONL file, with the format {"text": Story}.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode="a", encoding="utf-8") as f:
        writer = jsonlines.Writer(f)
        writer.write_all({"text": story} for story in stories)


def generate_stories(
    client,
    vocab: Vocabulary,
    model_name: str,
    total_requests: int,
    output_file: str,
    num_workers: int,
) -> None:
    """
    Generates stories using threads and saves them periodically.
    """
    start = time.time()
    stories_buffer = []
    success_count = 0
    failure_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_message, client, build_message(vocab), model_name)
            for _ in range(total_requests)
        ]

        for future in tqdm(futures, desc="Generating stories", total=total_requests):
            try:
                story = future.result(timeout=60)
                stories_buffer.append(story)
                success_count += 1
            except FuturesTimeoutError:
                pass
            except Exception as e:
                pass

            if len(stories_buffer) >= 50:
                save_stories_to_jsonl(stories_buffer, output_file)
                stories_buffer.clear()

    # Save any remaining stories
    if stories_buffer:
        save_stories_to_jsonl(stories_buffer, output_file)

    elapsed = time.time() - start
    print(f"{total_requests} stories generated in {elapsed:.2f} seconds.")


def generate_output_file(model_name: str, output_dir: str) -> str:
    """
    Generates a unique output file name based on the model name and current timestamp.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(
        output_dir,
        f"stories_{model_name}_{timestamp}.jsonl",
    )


def main(args):
    """
    Generates stories with Mistral's API and saves them to a JSONL file.
    """

    # Set the API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found. Set the MISTRAL_API_KEY environment variable."
        )

    # Initialize the vocabulary and client
    vocab = Vocabulary()
    client = get_client(api_key=api_key)
    output_file = generate_output_file(args.model_name, args.output_dir)

    # Generate and save stories
    generate_stories(
        client,
        vocab,
        args.model_name,
        args.total_requests,
        output_file,
        args.num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stories with a LLM.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistral-small-2503",
        help="Model name to use for generating stories.",
    )
    parser.add_argument(
        "--total_requests",
        type=int,
        default=512,
        help="Total number of requests to make.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="stories/mistral",
        help="Output directory for storing stories.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=3,
        help="Number of worker threads for parallel requests.",
    )
    args = parser.parse_args()
    main(args)
