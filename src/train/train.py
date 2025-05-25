import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import wandb
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
from utils import num_parameters, generate_text
from model_config import (
    ModelConfig_2M,
    ModelConfig_16M,
    ModelConfig_33M,
    ModelConfig_50M,
)
from train_config import TrainConfig


MODEL_CONFIG_CLASSES = {
    "2M": ModelConfig_2M,
    "16M": ModelConfig_16M,
    "33M": ModelConfig_33M,
    "50M": ModelConfig_50M,
}


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
        texts = [item["text"] + self.tokenizer.eos_token for item in batch]

        input_encodings = self.tokenizer(
            texts,
            max_length=self.block_size,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_encodings["labels"] = input_encodings["input_ids"].clone()
        input_encodings["labels"][:, :-1] = input_encodings["input_ids"][:, 1:]

        return input_encodings


def get_llama_config(config, tokenizer) -> LlamaConfig:
    """
    Convert the model configuration to LlamaConfig.
    """
    config_class = MODEL_CONFIG_CLASSES[config]
    config = config_class()
    return LlamaConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
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


def compute_batch_loss(model, batch, loss_fn, device):
    """
    Compute the loss for a batch of data.
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return loss


def evaluate(model, loss_fn, val_dataloader, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            loss = compute_batch_loss(model, batch, loss_fn, device)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    model.train()
    return avg_loss, perplexity.item()


def train(
    config,
    model,
    tokenizer,
    loss_fn,
    train_dataloader,
    val_dataloader,
    optimizer,
    lr_scheduler,
    device,
    output_dir,
):
    """
    Train the model on a single GPU.
    """
    # Training loop
    gradient_accumulation_steps = config.gradient_accumulation_steps
    completed_steps = 0
    start_context = "Il était une fois"
    best_val_loss = float("inf")
    pbar = tqdm(total=config.max_train_steps)

    model.train()
    for epoch in range(config.num_train_epochs):
        for step, batch in enumerate(train_dataloader, start=1):
            loss = compute_batch_loss(model, batch, loss_fn, device)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if step % 100 == 0:
                tqdm.write(
                    f"step {completed_steps} | loss/train: {loss.item() * gradient_accumulation_steps:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f}"
                )

            if step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                pbar.update(1)
                wandb.log(
                    {
                        "train_loss": loss.item() * gradient_accumulation_steps,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

            if (step % (config.eval_steps * gradient_accumulation_steps)) == 0:
                val_loss, perplexity = evaluate(model, loss_fn, val_dataloader, device)
                tqdm.write(f"loss/val: {val_loss:.4f} | perplexity: {perplexity:.2f}")
                wandb.log({"val_loss": val_loss, "perplexity": perplexity})

                # Generate text for evaluation
                generated_text = generate_text(
                    model,
                    tokenizer,
                    start_context,
                )
                print(f"Generated sample: {generated_text}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    tqdm.write(f"New best validation loss: {best_val_loss:.4f}")
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

        pbar.close()


def main(args):
    """
    Main function to train the Llama model on a single GPU.
    """
    # Train config
    train_config = TrainConfig()

    # Set the output directory
    output_dir = os.path.join(train_config.output_dir, args.model_config)

    # Make sure the Hugging Face token is set
    if not os.getenv("HF_TOKEN"):
        raise ValueError(
            "Please set the HF_TOKEN environment variable to your Hugging Face token."
        )

    # Initialize wandb
    wandb.init(project="LeCarnet", name="le-carnet-training-run")

    # Display training information
    print(f"Using device: {train_config.device}")
    print(f"Config: {args.model_config}")
    print(f"Tokenizer: {train_config.tokenizer_name}")
    print(f"Output directory: {output_dir}")

    # Load dataset and tokenizer
    train_dataset, val_dataset = get_dataset(
        train_config.dataset_name, train_config.cache_dir
    )
    print(
        f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
    )
    tokenizer = Tokenizer(train_config.tokenizer_name).get_tokenizer()
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Create dataloaders
    collate_fn = CollateFn(tokenizer, train_config.block_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_config.eval_batch_size,
        collate_fn=collate_fn,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )

    # Model
    llama_config = get_llama_config(args.model_config, tokenizer)
    model = LlamaForCausalLM(llama_config).to(train_config.device)

    # Define Loss, Optimizer and scheduler
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=train_config.num_warmup_steps,
        num_training_steps=train_config.max_train_steps,
    )

    print(f"Training {num_parameters(model) / 1e6:.2f}M parameters")
    print("Starting training...")
    train(
        train_config,
        model,
        tokenizer,
        loss_fn,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        train_config.device,
        output_dir,
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Llama-based model on a single GPU using LeCarnet dataset."
    )
    parser.add_argument(
        "--model_config",
        type=str,
        choices=["2M", "16M", "33M", "50M"],
        default="2M",
        help="Size of the model to train.",
    )
    args = parser.parse_args()
    main(args)
