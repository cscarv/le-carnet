import datasets as hfds
import json
from json import JSONDecodeError
from typing import List, Dict
from huggingface_hub import create_repo
import argparse


def load_validation_data(validation_path: str) -> List[Dict]:
    data = []
    with open(validation_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            try:
                record = json.loads(raw_line)
            except JSONDecodeError:
                continue
            data.append(record)
    return data


def create_parquet_dataset(file_path: str, repo_name: str, split: str = "validation"):
    data = load_validation_data(file_path)
    ds = hfds.Dataset.from_dict({"text": [item["text"] for item in data]})
    ds_dict = hfds.DatasetDict({split: ds})

    create_repo(repo_name, repo_type="dataset", private=True, exist_ok=True)
    ds_dict.push_to_hub(repo_name, private=True)


def main(args):

    create_parquet_dataset(
        file_path=args.file_path,
        repo_name=args.repo_name,
        split=args.split,
    )
    print("> Dataset uploaded!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a Parquet dataset from a JSONL file."
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="data/validation.jsonl",
        help="Path to the JSONL file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["validation", "train"],
        default="validation",
        help="Name of the split. Must be 'validation' or 'train'.",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default="MaxLSB/French-TinyStories",
        help="Name of the repository to create.",
    )
    args = parser.parse_args()
    main(args)
