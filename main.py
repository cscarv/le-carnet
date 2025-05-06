import os
import json
import sys
import torch
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader


def prepare_datasets(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split, cache_dir="./cache/")
    return dataset


def collate_fn(batch):
    return {"text": [item["text"] for item in batch]}


def get_dataloader(dataset, batch_size=16, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def translate_batch(texts, tokenizer, model, device, max_length=512):
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length
    )
    inputs = inputs.to(device)
    target_lang_token = tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_token,
            max_length=max_length,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translations_jsonl(loader, tokenizer, model, device, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(
            tqdm(
                loader,
                desc="Translating",
                unit="batch",
                file=sys.stdout,
                dynamic_ncols=True,
                leave=True,
            )
        ):
            texts = batch["text"]
            translations = translate_batch(texts, tokenizer, model, device)

            for t in translations:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Translate TinyStories with NLLB.")
    parser.add_argument(
        "--validation", action="store_true", help="Translate validation split"
    )
    parser.add_argument(
        "--training", action="store_true", help="Translate training split"
    )
    args = parser.parse_args()

    if not args.validation and not args.training:
        print("You must specify at least one of --validation or --training")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> Using device: {device}")
    model_name = "facebook/nllb-200-1.3B"

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "fra_Latn"
    tokenizer.padding_side = "right"
    print("> Tokenizer loaded.")

    # Load the model with 8-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    model.eval()
    print("> Model loaded.")

    # Load the dataset
    dataset_name = "roneneldan/TinyStories"
    if args.validation:
        dataset = prepare_datasets(dataset_name, "validation")
    else:
        dataset = prepare_datasets(dataset_name, "train")

    print(f"> Dataset loaded: {len(dataset)} samples")

    if args.validation:
        val_loader = get_dataloader(dataset, batch_size=32, num_workers=4)
        print("> Translating validation dataset")
        translations_jsonl(
            val_loader, tokenizer, model, device, output_file="data/val.jsonl"
        )
        print("> Saved validation translations → data/val.jsonl")

    if args.training:
        train_loader = get_dataloader(dataset, batch_size=32, num_workers=4)
        print("> Translating training dataset")
        translations_jsonl(
            train_loader,
            tokenizer,
            model,
            device,
            output_file="data/train.jsonl",
        )
        print("> Saved training translations → data/train.jsonl")


if __name__ == "__main__":
    main()
