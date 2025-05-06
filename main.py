import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Translate TinyStories with NLLB.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation"],
        required=True,
        help="Dataset splits to translate (train, validation).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for DataLoader."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="Directory for output files.",
    )
    return parser.parse_args()


def collate_fn(batch):
    return {"text": [item["text"] for item in batch]}


def get_dataloader(dataset, batch_size: int, num_workers: int):
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
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    ).to(device)

    target_lang_id = tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_length=max_length,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_split(
    split_name: str,
    dataset_name: str,
    tokenizer,
    model,
    device,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
):
    split = "validation" if split_name == "validation" else "train"
    dataset = load_dataset(dataset_name, split=split, cache_dir="./cache/")
    print(f"> Loaded {split} split: {len(dataset)} samples")

    loader = get_dataloader(dataset, batch_size, num_workers)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{split}.jsonl"

    # Check if the output file already exists
    if out_file.exists():
        answer = input(f"File '{out_file}' exists. Overwrite? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print(f"> Skipping {split} split translation.")
            return

    with out_file.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc=f"Translating {split}", unit="batch"):
            translated = translate_batch(batch["text"], tokenizer, model, device)
            for text in translated:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"> Saved {split} translations to {out_file}")


def main():
    args = parse_args()
    model_name = "facebook/nllb-200-1.3B"
    dataset_name = "roneneldan/TinyStories"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "fra_Latn"
    tokenizer.padding_side = "right"
    print(f"> Loaded tokenizer: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        # cache_dir="./cache/",
    )
    model = torch.compile(model)
    print(f"> Loaded model: {model_name}")
    model.eval()

    for split_name in args.splits:
        translate_split(
            split_name,
            dataset_name,
            tokenizer,
            model,
            device,
            args.batch_size,
            args.num_workers,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
