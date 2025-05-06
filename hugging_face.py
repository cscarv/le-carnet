import datasets as hfds
import json
from typing import List, Dict


def load_validation_data(validation_path: str) -> List[Dict]:
    data = []
    with open(validation_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_parquet_dataset(validation_path: str):
    data = load_validation_data(validation_path)
    dataset_dict = {
        "text": [item["text"] for item in data],
    }

    dataset = hfds.Dataset.from_dict(dataset_dict)

    dataset_dict_split = hfds.DatasetDict(
        {"validation": dataset}
    )  # change to train if needed

    dataset_dict_split.save_to_disk("data/validation_parquet")
    dataset_dict_split.push_to_hub("MaxLSB/TinyStories-fr", private=True)


def main():

    validation_path = "data/validation.jsonl"  # change to train if needed
    create_parquet_dataset(validation_path)

    print("Dataset saved as Parquet and uploaded to Hugging Face!")


if __name__ == "__main__":
    main()
