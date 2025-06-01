import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import platform
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
import jsonlines


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


def translate_batch(texts, tokenizer, model, device, max_length):
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
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_split(
    split: str,
    dataset_name: str,
    tokenizer,
    model,
    device,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
    max_length: int,
):
    dataset = load_dataset(dataset_name, split=split)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{split}.jsonl"

    start_idx = 0
    if out_file.exists():
        with jsonlines.open(out_file, mode="r") as reader:
            start_idx = sum(1 for _ in reader)
        print(f"Resuming from sample #{start_idx}")

    if start_idx >= len(dataset):
        print(f"All {len(dataset)} samples already processed—skipping.")
        return

    dataset = dataset.select(range(start_idx, len(dataset)))
    loader = get_dataloader(dataset, batch_size, num_workers)

    with open(out_file, mode="a", encoding="utf-8") as f:
        writer = jsonlines.Writer(f)
        for batch in tqdm(loader, desc=f"Translating {split}", unit="batch"):
            translated = translate_batch(
                batch["text"], tokenizer, model, device, max_length
            )
            for text in translated:
                writer.write({"text": text})

    print(f"Saved {len(dataset)} more translations to {out_file}")


def main(args):
    """
    Main function to load the model and tokenizer, and translate the dataset.
    """
    dataset_name = "roneneldan/TinyStories"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "fra_Latn"
    tokenizer.padding_side = "right"
    print(f"Loaded tokenizer: {args.model_name}")

    quantization_config = None
    if platform.system() != "Darwin":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        print("Using bitsandbytes 8-bit quantization")
    else:
        print("Skipping bitsandbytes quantization")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )
    model = torch.compile(model)
    print(f"Loaded model: {args.model_name}")
    model.eval()

    translate_split(
        args.split,
        dataset_name,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.num_workers,
        args.output_dir,
        args.max_length,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate TinyStories with NLLB.")
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["train", "validation"],
        required=True,
        help="Dataset splits to translate (train, validation).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        choices=[
            "facebook/nllb-200-distilled-600M",
            "facebook/nllb-200-distilled-1.3B",
            "facebook/nllb-200-1.3B",
            "facebook/nllb-200-3.3B",
        ],
        help="Model name to use for translation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for DataLoader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers. Increase if more CPU cores are available.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max length for tokenization and generation.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("stories/translations"),
        help="Directory for output files.",
    )
    args = parser.parse_args()
    main(args)
