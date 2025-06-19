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
    """
    A class to load and manage vocabulary for story generation.
    """
    def __init__(self, vocab_dir="src/data/vocabulary"):
        categories = ["adjectives", "nouns", "verbs", "features"]
        for category in categories:
            file_path = os.path.join(vocab_dir, f"{category}.txt")
            with open(file_path, "r") as f:
                setattr(self, category, [line.strip() for line in f if line.strip()])

    def random_choice(self, category):
        return random.choice(getattr(self, category))


def build_message(vocab: Vocabulary):
    """
    Builds a message for the LLM to generate a story.
    """
    verb = vocab.random_choice("verbs")
    noun = vocab.random_choice("nouns")
    adjective = vocab.random_choice("adjectives")
    feature = vocab.random_choice("features")

    prompt = f"""
    Écris une courte histoire en français adaptée à des enfants de 5 à 7 ans.
    Utilise des mots simples et faciles à comprendre.
    L'histoire doit rester logique et cohérente.
    Limite l'histoire à 2 ou 3 courts paragraphes (environ 100 à 150 mots).
    Intègre naturellement le verbe «{verb}», le nom «{noun}» et l’adjectif «{adjective}».
    L’histoire doit avoir la caractéristique suivante : {feature}.
    N’oublie pas d’utiliser uniquement des mots simples et de garder l’histoire courte !
    """

    message = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    return message


def get_client(api_key: str):
    """
    Initializes the Mistral client with the provided API key.
    """
    return Mistral(api_key=api_key)


def send_message(client, message, model_name):
    """
    Sends a message to the Mistral API with retries and exponential backoff.
    Adds a small random delay to stagger requests.
    """
    time.sleep(random.uniform(0.1, 0.5))  # To avoid rate limit spikes
    backoff = 1.0
    for attempt in range(5):
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

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_message, client, build_message(vocab), model_name)
            for _ in range(total_requests)
        ]

        for future in tqdm(futures, desc="Generating stories", total=total_requests):
            try:
                story = future.result(timeout=60)
                stories_buffer.append(story)
            except FuturesTimeoutError:
                pass
            except Exception as e:
                pass
            
            # Saving stories by batch of 100
            if len(stories_buffer) >= 100:
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

    # Initialize the vocabulary, client and output file
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
        default="mistral-large-2411",
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
        default=4,
        help="Number of worker threads for parallel requests.",
    )
    args = parser.parse_args()
    main(args)
