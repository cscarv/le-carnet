import argparse
import os
import torch
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    get_scheduler,
)

from config import ModelConfig_3M


class Tokenizer:
    """
    Tokenizer class to handle tokenization and padding.
    """

    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = "right"

    def get_tokenizer(self):
        return self.tokenizer


def get_dataset(dataset_name, cache_dir):
    train_dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    val_dataset = load_dataset(dataset_name, split="validation", cache_dir=cache_dir)
    return train_dataset, val_dataset


class CollateFn:
    def __init__(self, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __call__(self, batch):
        texts = [item["text"] for item in batch]

        input_encodings = self.tokenizer(
            texts,
            max_length=self.block_size,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_encodings["labels"] = input_encodings["input_ids"].clone()
        input_encodings["labels"][:, :-1] = input_encodings["input_ids"][:, 1:]
        input_encodings["labels"][:, -1] = self.tokenizer.eos_token_id

        return input_encodings


def get_llama_config(config) -> LlamaConfig:
    """
    Convert the model configuration to LlamaConfig.
    """
    return LlamaConfig(
        vocab_size=config.vocab_size,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        initializer_range=config.initializer_range,
        hidden_act=config.hidden_act,
        tie_word_embeddings=config.tie_word_embeddings,
    )


def get_hf_repo(repo_name: str, output_dir: str) -> Repository:
    """
    Create or connect to a Hugging Face repository for the model.
    """
    create_repo(repo_name, exist_ok=True, private=True)
    repo = Repository(
        local_dir=output_dir,
        clone_from=repo_name,
        use_auth_token=True,
    )
    return repo


def evaluate(model, eval_dataloader, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
    avg_loss = total_loss / len(eval_dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()


def num_parameters(model):
    """
    Count the number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    args,
    model,
    tokenizer,
    train_dataloader,
    eval_dataloader,
    optimizer,
    lr_scheduler,
    device,
    repo,
):
    """
    Train the model on a single GPU.
    """

    # Training loop
    gradient_accumulation_steps = args.gradient_accumulation_steps
    completed_steps = 0

    model.train()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(
            tqdm(train_dataloader, total=args.max_train_steps), start=1
        ):

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if step % 100 == 0:
                tqdm.write(
                    f"> step {step} | loss/train: {loss.item():.4f} | lr: {optimizer.param_groups[0]['lr']:.6f}"
                )

            if step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

            if (step % (args.eval_steps * gradient_accumulation_steps)) == 0:
                eval_loss, perplexity = evaluate(model, eval_dataloader, device)
                tqdm.write(
                    f"> loss/eval: {eval_loss:.4f} | perplexity: {perplexity:.2f}"
                )
                model.train()

                # Save model and tokenizer to the Hugging Face Hub and output directory
                os.makedirs(args.output_dir, exist_ok=True)
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress step {step}", blocking=False
                )

        # Save model
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


def main(args):
    """
    Main function to train the Llama model on a single GPU.
    """
    # Setting the HF repo
    repo = get_hf_repo(args.repo_name, args.output_dir)

    # Setup the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and tokenizer
    train_dataset, val_dataset = get_dataset(args.dataset_name, args.cache_dir)
    tokenizer = Tokenizer(args.tokenizer_name).get_tokenizer()
    block_size = args.block_size

    # Create dataloaders
    collate_fn = CollateFn(tokenizer, block_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn
    )

    # Model configuration
    config = ModelConfig_3M()
    llama_config = get_llama_config(config)
    model = LlamaForCausalLM(llama_config).to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    print(f"> Training {num_parameters(model) / 1e6:.2f}M parameters")

    train(
        args,
        model,
        tokenizer,
        train_dataloader,
        eval_dataloader,
        optimizer,
        lr_scheduler,
        device,
        repo,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Llama-based model on a single GPU."
    )
    parser.add_argument("--repo_name", type=str, default="MaxLSB/LeCarnet-3M")
    parser.add_argument("--dataset_name", type=str, default="MaxLSB/LeCarnet")
    parser.add_argument("--tokenizer_name", type=str, default="openai-community/gpt2")
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--cache_dir", type=str, default="cache/")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=None)
    args = parser.parse_args()
    main(args)
