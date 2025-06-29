import time
import random
import os
import re
import json
import numpy as np
from datasets import load_dataset
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from httpx import RemoteProtocolError
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_client(api_key: str) -> Mistral:
    """Initialize and return a Mistral API client."""
    return Mistral(api_key=api_key)


def get_evaluation_response(client: Mistral, message: str, model_name: str) -> str:
    """Send evaluation prompts to the API and return the response."""
    message = message.replace("\n", " ").strip()
    prompt1 = """
    In the following exercise, the student is given the beginning of a story. They must complete it to make a full story.
    The exercise evaluates the student's language skills and creativity. The symbol *** marks the separation between the provided beginning and the student's continuation.

    Here is the story:
    {message}

    Provide an overall assessment of the part written by the student (the part after the *** symbol).
    Is it grammatically correct? Is it coherent with the beginning of the story?
    Pay particular attention to how well the student manages to complete the sentence that was interrupted at the *** separator.
    """.format(
        message=message
    )

    prompt2 = """
    Now give a score from 0 to 10 to the student for each of the following categories: grammar, creativity, coherence with the beginning of the story, and logical flow of the plot.

    You must respond using the following format:

    <Grammar>[score]</Grammar>
    <Creativity>[score]</Creativity>
    <Coherence>[score]</Coherence>
    <Logic>[score]</Logic>
    """

    resp1 = client.chat.complete(
        messages=[{"role": "user", "content": prompt1}],
        model=model_name,
        temperature=0.5,
        max_tokens=512,
    )
    reply1 = resp1.choices[0].message.content

    full_messages = [
        {"role": "user", "content": prompt1},
        {"role": "assistant", "content": reply1},
        {"role": "user", "content": prompt2},
    ]
    time.sleep(1)
    resp2 = client.chat.complete(
        messages=full_messages,
        model=model_name,
        temperature=0.5,
        max_tokens=512,
    )
    return resp2.choices[0].message.content


def extract_grades(response: str) -> list[int]:
    """Extract numerical grades from the API response using tags."""
    tags = ["Grammar", "Creativity", "Coherence", "Logic"]
    grades = []
    for tag in tags:
        match = re.search(rf"<{tag}>\s*(\d+)\s*</{tag}>", response)
        if not match:
            raise ValueError(f"Grade for {tag} not found in the response.")
        grade = int(match.group(1))
        if not 0 <= grade <= 10:
            raise ValueError(f"Grade for {tag} is out of range (0-10): {grade}")
        grades.append(grade)
    return grades


def eval_story(client: Mistral, message: str, model_name: str) -> list[int]:
    """Evaluate a story and return grades, with retry logic."""
    for attempt in range(3):
        if attempt > 0:
            print(f"Retry {attempt}/3")
        try:
            response = get_evaluation_response(client, message, model_name)
            return extract_grades(response)
        except (SDKError, RemoteProtocolError, ValueError) as e:
            if (
                isinstance(e, SDKError)
                and getattr(e, "status_code", None) == 429
                and attempt < 2
            ):
                time.sleep(1 + random.random() * 0.5)
            elif attempt < 2:
                time.sleep(1 + random.random() * 0.5)
            else:
                raise e
    raise RuntimeError("Max retries reached! Unable to evaluate the story.")


def generate_completions(model, tokenizer, prompts, device, batch_size=2) -> list[str]:
    """Generate story completions for a list of prompts in batches."""
    completions = []
    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc="Generating completions (batches)",
        total=(len(prompts) + batch_size - 1) // batch_size,
    ):
        base_batch_prompts = prompts[i : i + batch_size]
        # Remove the ending '***' from each prompt for generation
        batch_prompts = [
            prompt.split("***")[0].rstrip() for prompt in base_batch_prompts
        ]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=512,
            temperature=0.1,
            top_k=50,
            do_sample=True,
            no_repeat_ngram_size=4,
        )
        for j in range(len(batch_prompts)):
            decoded = tokenizer.decode(outputs[j], skip_special_tokens=True)
            generated = decoded[len(batch_prompts[j]) :].strip()
            full_text = base_batch_prompts[j] + generated
            completions.append(full_text)
    return completions


# Function to load the dataset
def get_dataset(name: str, split: str = "test", cache_dir: str = None):
    """Load and return the dataset."""
    return load_dataset(name, split=split, cache_dir=cache_dir)


def get_tokenizer(model_name: str):
    """Load and return the tokenizer for the specified model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def save_results_to_json(
    output_dir: str,
    model_name: str,
    judge_model_name: str,
    grades_dict: dict[str, float],
) -> None:
    """Save the evaluation results to a JSON file."""
    filename = f"{model_name.replace('/', '_')}.json"
    out_path = os.path.join(output_dir, filename)
    data = {}
    if os.path.isfile(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[judge_model_name] = grades_dict
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


def main(args):
    # Get API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found. Set the MISTRAL_API_KEY environment variable."
        )

    # Load model and tokenizer
    tokenizer = get_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
    dataset = get_dataset(args.dataset_name, split="test", cache_dir="cache/")

    # Extract prompts
    prompts = [example["text"] for example in dataset]

    # Generate completions in batches
    completions = generate_completions(
        model, tokenizer, prompts, args.device, batch_size=args.batch_size
    )

    # Evaluate stories in parallel
    grades = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_index = {
            executor.submit(
                eval_story, get_client(api_key), full_text, args.judge_model_name
            ): i
            for i, full_text in enumerate(completions)
        }

        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc="Evaluating stories",
        ):
            index = future_to_index[future]
            try:
                grade = future.result()
                grades.append((index, grade))
            except Exception as e:
                raise RuntimeError(f"Evaluation for story {index} failed: {e}")

    # Sort grades by original index and extract them
    grades.sort(key=lambda x: x[0])
    grades = [grade for _, grade in grades]

    # Check if any evaluations succeeded
    if not grades:
        raise RuntimeError("No stories were successfully evaluated.")

    print(f"Correctly evaluated {len(grades)} stories out of {len(completions)} total.")

    # Compute and display final grades
    final_grade = np.mean(grades, axis=0).tolist()
    print("Final Grades (Grammar, Creativity, Coherence, Logic):", final_grade)

    # Save results to JSON
    grades_dict = {
        "Grammar": final_grade[0],
        "Creativity": final_grade[1],
        "Coherence": final_grade[2],
        "Logic": final_grade[3],
    }
    os.makedirs(args.output_dir, exist_ok=True)
    save_results_to_json(
        args.output_dir,
        args.model_name,
        args.judge_model_name,
        grades_dict,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for LeCarnet models"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="MaxLSB/LeCarnet-3M",
        help="Model name to evaluate (HF repo).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MaxLSB/LeCarnet",
        help="Dataset name to use for evaluation (HF repo).",
    )
    parser.add_argument(
        "--judge_model_name",
        type=str,
        default="mistral-large-2411",
        help="Judge model name to use for evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory where evaluation JSON files will be saved.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for parallel evaluation.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for story generation."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on.",
    )
    args = parser.parse_args()
    main(args)
