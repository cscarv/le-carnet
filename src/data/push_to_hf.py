import datasets as hfds
import json
from json import JSONDecodeError
from typing import List, Dict
from huggingface_hub import create_repo
import argparse
import os


def load_data(file_path: str) -> List[Dict]:
    """
    Load data from a JSONL file.
    Each line in the file should be a valid JSON object.
    """
    data = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            try:
                record = json.loads(line)
                data.append(record)
            except JSONDecodeError as e:
                print(f"Line {line_number}: JSONDecodeError: {e}")
                continue
    return data


def clean_data(data: List[Dict]) -> List[Dict]:
    """
    Clean the data by removing duplicates and replacing special characters.
    """
    seen = set()
    cleaned = []
    for item in data:
        text = item["text"].replace("Ã¢", "â").replace("Ã©", "é")
        if text not in seen:
            seen.add(text)
            cleaned.append({"text": text})
    return cleaned


def create_parquet_dataset(folder_path: str, repo_name: str):
    """
    Create a dataset from JSONL files in the specified folder and push it to the Hugging Face Hub.
    Each JSONL file should be named according to the split (e.g., train.jsonl, validation.jsonl).
    """
    dataset_splits = {}

    for file in os.listdir(folder_path):
        if file.endswith(".jsonl"):
            split_name = os.path.splitext(file)[0]
            file_path = os.path.join(folder_path, file)

            data = load_data(file_path)
            print(f"Loaded {len(data)} records from {file_path}")
            data = clean_data(data)
            print(f"Cleaned to {len(data)} records")

            ds = hfds.Dataset.from_dict({"text": [item["text"] for item in data]})
            dataset_splits[split_name] = ds

    ds_dict = hfds.DatasetDict(dataset_splits)
    create_repo(repo_name, repo_type="dataset", private=True, exist_ok=True)
    ds_dict.push_to_hub(repo_name, private=True)

    print(f"\nSuccessfully pushed the following splits to the Hub repo '{repo_name}':")
    for split, ds in ds_dict.items():
        print(f"{split}: {len(ds)} samples")


def main(args):
    # Make sure the jsonl files in the folder have the correct names 'train.jsonl' and/or 'validation.jsonl'

    print(f"Pushing the dataset to the Hugging Face Hub repo '{args.repo_name}'...")

    create_parquet_dataset(
        folder_path=args.folder_path,
        repo_name=args.repo_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the dataset (folder of json files) to parquet and push it to the Hugging Face Hub."
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        default="backup/",
        help="Path to the folder containing the JSONL files.",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default="MaxLSB/LeCarnet",
        help="Name of the repository to create.",
    )
    args = parser.parse_args()
    main(args)
