import os
from tqdm import tqdm
import time
import random
import jsonlines
import argparse
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError


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
    feature_1, feature_2 = vocab.random_features(2)

    prompt = f"""
    Écris une courte histoire (3 à 5 paragraphes) qui n’utilise que des mots très simples qu’un enfant de 3 ans comprendrait. 
    L’histoire doit utiliser le verbe « {verb} », le nom « {noun} » et l’adjectif « {adjective} ». 
    L'histoire doit avoir les caractéristiques suivantes : {feature_1}, {feature_2}. 
    Souviens-toi de n’utiliser que des mots simples !"
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
    backoff = 1.0
    for attempt in range(5):  # 5 attempts total
        try:
            resp = client.chat.complete(
                messages=message,
                model=model_name,
                temperature=0.8,
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
) -> list[dict]:
    """
    Generates stories using the specified model and saves them to a JSONL file every 50 stories.
    """
    stories = []
    start = time.time()

    for _ in tqdm(range(total_requests), desc="Generating stories"):
        message = build_message(vocab)
        story = send_message(
            client,
            message,
            model_name=model_name,
        )

        stories.append(story)

        if len(stories) % 50 == 0:  # Save every 50 stories
            save_stories_to_jsonl(stories, output_file)
            stories = []

    # Save any remaining stories
    if stories:
        save_stories_to_jsonl(stories, output_file)

    elapsed = time.time() - start
    tqdm.write(f"{total_requests} stories generated in {elapsed:.2f} seconds.")


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
            f"API key not found. Set the MISTRAL_API_KEY environment variable."
        )

    # Initialize the vocabulary and client
    vocab = Vocabulary()
    client = get_client(api_key=api_key)
    output_file = generate_output_file(args.model_name, args.output_dir)

    # Generate batches of stories
    generate_stories(
        client,
        vocab,
        args.model_name,
        args.total_requests,
        output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stories with a LLM.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistral-small-2501",
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
        default="results/generated/mistral",
        help="Output file path for storing stories.",
    )
    args = parser.parse_args()
    main(args)
