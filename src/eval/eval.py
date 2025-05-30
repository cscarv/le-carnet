import time
import random
from datasets import load_dataset
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from httpx import RemoteProtocolError
import os
import re
import numpy as np
from tqdm import tqdm


def get_client(api_key: str):
    return Mistral(api_key=api_key)


def send_message(client, message1, message2, model_name):
    backoff = 1.0

    for attempt in range(5):
        try:

            resp1 = client.chat.complete(
                messages=[{"role": "user", "content": message1}],
                model=model_name,
                temperature=0.8,
                max_tokens=512,
                top_p=0.95,
            )

            reply1 = resp1.choices[0].message.content

            full_messages = [
                {"role": "user", "content": message1},
                {"role": "assistant", "content": reply1},
                {"role": "user", "content": message2},
            ]

            resp2 = client.chat.complete(
                messages=full_messages,
                model=model_name,
                temperature=0.8,
                max_tokens=512,
                top_p=0.95,
            )

            return resp2.choices[0].message.content

        except (SDKError, RemoteProtocolError) as e:
            if attempt < 4:
                wait = backoff + random.random() * 0.5
                time.sleep(wait)
                backoff *= 2
            else:
                raise e

    raise RuntimeError("Max retries reached")


def extract_grades(prompt):
    notes = re.findall(r"\b(\d{1,2})/10\b", prompt)

    return [int(note) for note in notes]


def eval_story(
    client,
    message,
    model_name: str,
) -> list[dict]:
    """
    Generates evaluations using the specified model and returns the grades in order (Grammar, Creativity, Coherence, Logic).
    """

    message = message.replace("\n", " ").strip()

    mistral_prompt_1 = """
        Dans l'exercice suivant, l'élève reçoit un début d'histoire. Il doit le compléter pour en faire une histoire complète.
        L'exercice évalue les compétences linguistiques et la créativité de l'élève. Le symbole *** marque la séparation entre le début imposé et la suite rédigée par l'élève.

        Voici l'histoire :
        {message}
        
        Fourni une évaluation générale de la partie rédigée par l'élève (celle qui se trouve après le symbole ***).
        Est-elle grammaticalement correcte ? Est-elle cohérente avec le début de l'histoire ?
        Porte une attention particulière à la façon dont l'élève parvient à compléter la phrase interrompue au milieu par le séparateur ***.
    """

    mistral_prompt_2 = """
        Maintenant donne une note de 0 à 10 à l’élève pour chaque catégorie : grammaire, créativité, cohérence avec le début de l’histoire et logique du déroulement de l’intrigue.
        De plus, donne ta meilleure estimation de l’âge de l’élève tel qu’il ressort de sa rédaction.
        Choisissez parmi les groupes d’âge suivants :
        A : 3 ans ou moins
        B : 4-5 ans
        C : 6-7 ans
        D : 8-9 ans
        E : 10-12 ans
        F : 13-16 ans
    """

    for attempt in range(3):
        try:
            evaluation = send_message(
                client,
                mistral_prompt_1.format(message=message),
                mistral_prompt_2,
                model_name=model_name,
            )

            # print(f"Evaluation received: {evaluation}")

            grade = extract_grades(evaluation)
            return grade
        except SDKError as e:
            if getattr(e, "status_code", None) == 429 and attempt < 2:
                wait = 1.0 + random.random() * 0.5
                time.sleep(wait)
            else:
                raise e

    raise RuntimeError("Max retries reached")


def get_dataset(dataset_name, split="test", cache_dir=None):
    """
    Load the specified split of the dataset from the Hugging Face Hub.
    """
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    return dataset


def tokenize_prompt(prompt, tokenizer):
    """
    Tokenizes the input prompt using the provided tokenizer.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(args.device)
    return input_ids


def main(args):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
    df = get_dataset(args.dataset_name, split="test", cache_dir="cache/")

    grades = np.zeros(
        (len(df), 4), dtype=int
    )  # Assuming 4 categories: Grammar, Creativity, Coherence, Logic

    for i, prompt in enumerate(tqdm(df["text"], desc="Evaluating stories")):
        inputs = tokenizer(prompt[:-3], return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)

        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            temperature=0.1,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        output = model.generate(**generation_kwargs)

        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_part = decoded_output[len(prompt[:-3]) :]

        eval_part = prompt + generated_part

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                f"API key not found. Set the MISTRAL_API_KEY environment variable."
            )

        client = get_client(api_key=api_key)

        grade = eval_story(client, eval_part, args.eval_model_name)

        if len(grade) != 4:
            print(
                f"Error: Expected 4 grades, got {len(grade)} for prompt {i}. Skipping this entry."
            )
            continue

        grades[i] = grade

    final_grade = np.mean(grades, axis=0)
    print("Final Grades (Grammar, Creativity, Coherence, Logic):", final_grade)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for LeCarnet models"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="MaxLSB/LeCarnet-3M",
        help="Model name to evaluate.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MaxLSB/LeCarnet",
        help="dataset_name to use for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda or cpu).",
    )
    parser.add_argument(
        "--eval_model_name",
        type=str,
        default="mistral-small-2503",
        help="Model name to use for evaluation.",
    )
    args = parser.parse_args()
    main(args)
