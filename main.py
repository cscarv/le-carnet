from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import json
import torch
from tqdm import tqdm


def prepare_datasets(dataset_name):
    train_dataset = load_dataset(
        dataset_name,
        split="train",
        cache_dir="./cache/",
    )

    val_dataset = load_dataset(
        dataset_name,
        split="validation",
        cache_dir="./cache/",
    )
    return train_dataset, val_dataset


def collate_fn(batch):
    return {
        "text": [f">>fra<< {item['text']}" for item in batch],
    }


def get_dataloader(dataset, batch_size=32, num_workers=4):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return dataloader


def translate_texts(text, model, tokenizer, device):
    encoded = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    translated = model.generate(
        **encoded,
        max_length=512,
        num_beams=4,
        early_stopping=False,
        do_sample=True,
        temperature=0.7,
        top_k=50,
    )

    return tokenizer.batch_decode(translated, skip_special_tokens=True)


def translations_jsonl(
    train_loader,
    model,
    tokenizer,
    device,
    output_file="data/train.jsonl",
    dump_every: int = 10,
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    buffer = []
    with open(output_file, "w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc="Translating", unit="batch")
        ):
            texts = batch["text"]
            translations = translate_texts(texts, model, tokenizer, device)
            for t in translations:
                buffer.append(json.dumps({"text": t}, ensure_ascii=False) + "\n")

            # every `dump_every` batches, flush buffer to disk
            if (batch_idx + 1) % dump_every == 0:
                f.writelines(buffer)
                f.flush()
                buffer.clear()

        # write any remaining records
        if buffer:
            f.writelines(buffer)
            f.flush()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"> Device: {device}")

    dataset_name = "roneneldan/TinyStories"
    train_dataset, val_dataset = prepare_datasets(dataset_name)
    print(
        f"> Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples"
    )

    model_name = "Helsinki-NLP/opus-mt-en-fr"
    model = MarianMTModel.from_pretrained(model_name).to(device)
    print(f"> Model loaded:: {model_name}")

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    print(f"> Tokenizer loaded")

    # Translating the train set
    val_loader = get_dataloader(val_dataset, batch_size=64, num_workers=4)
    print(f"> Val DataLoader created")

    translations_jsonl(
        val_loader,
        model,
        tokenizer,
        device,
        output_file="data/val.jsonl",
    )
    print(f"> Validation translations saved to data/val.jsonl")

    # Translating train set
    train_loader = get_dataloader(train_dataset, batch_size=64, num_workers=4)
    print(f"> Train DataLoader created")

    translations_jsonl(
        train_loader,
        model,
        tokenizer,
        device,
        output_file="data/train.jsonl",
    )
    print(f"> Train translations saved to data/train.jsonl")


if __name__ == "__main__":
    main()
