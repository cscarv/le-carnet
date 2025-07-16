import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tok = "hf" + "." + "tokens"
with open(tok, "r") as f:
    hf_token = f.read().strip()
os.environ["HF_TOKEN"] = hf_token
import argparse
import torch
import wandb
import math
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
    get_scheduler,
)
from utils import (
    num_parameters,
    generate_text,
    get_amp_scaler_and_autocast,
    get_mixed_precision_dtype,
)
from configs import (
    TrainConfigEnglish,
    TrainConfigFrench,
    CustomConfig,
    ModelConfig_3M,
    ModelConfig_8M,
    ModelConfig_21M,
)


MODEL_CONFIG_CLASSES = {
    "custom": CustomConfig,
    "3M": ModelConfig_3M,
    "8M": ModelConfig_8M,
    "21M": ModelConfig_21M,
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
        input_encodings["labels"][:, -1] = self.tokenizer.pad_token_id

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


def compute_batch_loss(model, batch, loss_fn, device, autocast_ctx):
    """
    Compute the loss for a batch of data.
    """
    with autocast_ctx:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return loss


def evaluate(model, loss_fn, val_dataloader, device, autocast_ctx):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            loss = compute_batch_loss(model, batch, loss_fn, device, autocast_ctx)
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
    scaler,
    autocast_ctx,
):
    """
    Train the model on a single GPU.
    """
    # Load checkpoint if exists
    if config.load_checkpoint and os.path.exists(config.load_checkpoint_path):
        tqdm.write(f"Loading checkpoint from {config.load_checkpoint_path}")
        checkpoint = torch.load(config.load_checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if scaler is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        effective_steps = checkpoint["effective_steps"]
        best_val_loss = checkpoint["best_val_loss"]
        tqdm.write(
            f"Checkpoint loaded! Resuming from epoch {start_epoch}, effective_steps {effective_steps}"
        )
    else:
        start_epoch = 0
        effective_steps = 0
        best_val_loss = float("inf")

    gradient_accumulation_steps = config.gradient_accumulation_steps
    start_context = "Il était une fois"
    pbar = tqdm(total=config.total_iterations, initial=effective_steps, desc="Training")
    model.train()

    for epoch in range(start_epoch, config.num_epochs):
        for step, batch in enumerate(train_dataloader, start=1):
            loss = compute_batch_loss(
                model, batch, loss_fn, config.device, autocast_ctx
            )
            loss = loss / gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % 100 == 0:
                tqdm.write(
                    f"step {effective_steps} | loss/train: {loss.item() * gradient_accumulation_steps:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f}"
                )

            if step % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()
                effective_steps += 1
                pbar.update(1)
                wandb.log(
                    {
                        "train_loss": loss.item() * gradient_accumulation_steps,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

            if (step % (config.eval_steps * gradient_accumulation_steps)) == 0:
                val_loss, perplexity = evaluate(
                    model, loss_fn, val_dataloader, config.device, autocast_ctx
                )
                tqdm.write(f"loss/val: {val_loss:.4f} | perplexity: {perplexity:.2f}")
                wandb.log({"val_loss": val_loss, "perplexity": perplexity})

                # Generate text for evaluation
                generated_text = generate_text(
                    model,
                    tokenizer,
                    start_context,
                )
                tqdm.write(f"Generated text: {generated_text}")

                # Save the best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    tqdm.write(f"New best validation loss: {best_val_loss:.4f}")
                    os.makedirs(config.output_dir + "model_weights/", exist_ok=True)
                    model.save_pretrained(config.output_dir + "model_weights/")
                    tokenizer.save_pretrained(config.output_dir + "model_weights/")

        # Save checkpoint at the end of each epoch
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "effective_steps": effective_steps,
            "best_val_loss": best_val_loss,
        }
        os.makedirs(config.output_dir + "checkpoints/", exist_ok=True)
        torch.save(
            checkpoint, config.output_dir + f"checkpoints/checkpoint-epoch-{epoch}.pt"
        )
        tqdm.write(
            f"Checkpoint saved at epoch {epoch}, effective_steps {effective_steps}"
        )

    pbar.close()
    tqdm.write("Training complete.")


def main(args):
    """
    Main function to train the Llama model on a single GPU.
    """
    # Train config
    if args.language == "english":
        train_config = TrainConfigEnglish()
    elif args.language == "french":
        train_config = TrainConfigFrench()
    else:
        raise ValueError("Language must be either 'english' or 'french'.")

    # Make sure the Hugging Face token is set
    if not os.getenv("HF_TOKEN"):
        raise ValueError(
            "Please set the HF_TOKEN environment variable to your Hugging Face token."
        )

    # Initialize wandb
    wandb.init(project="multi-teacher-distillation", name="english-teacher-3m" if args.language == "english" else "french-teacher-3m")

    tqdm.write("Loading dataset and tokenizer...")
    # Load dataset and tokenizer
    # train_dataset, val_dataset = get_dataset(
    #     train_config.dataset_name, train_config.cache_dir
    # )
    train_dataset = load_from_disk(train_config.train_dataset_path)
    val_dataset = load_from_disk(train_config.val_dataset_path)
    # tokenizer = Tokenizer(train_config.tokenizer_name).get_tokenizer()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=train_config.tokenizer_path)
    special_tokens_dict = {
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]"
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    print("EOS token:", tokenizer.eos_token)

    # Display training information
    tqdm.write(f"Using device: {train_config.device}")
    tqdm.write(f"Config: {args.model_config}")
    tqdm.write(f"Tokenizer: {train_config.tokenizer_path}")
    tqdm.write(f"Output directory: {train_config.output_dir}")
    tqdm.write(
        f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
    )
    if train_config.mixed_precision:
        tqdm.write(f"Using mixed precision: {get_mixed_precision_dtype()}")

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

    # Compute total iterations for num_epochs
    train_config.total_iterations = math.ceil(
        len(train_dataloader)
        * train_config.num_epochs
        / train_config.gradient_accumulation_steps
    )

    # Define Loss, Optimizer and scheduler
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=train_config.num_warmup_steps,
        num_training_steps=train_config.total_iterations,
    )

    # Get AMP scaler and autocast context for mixed precision training
    scaler, autocast_ctx = get_amp_scaler_and_autocast(
        train_config.device, train_config.mixed_precision
    )

    tqdm.write(f"Training {num_parameters(model) / 1e6:.2f}M parameters")
    tqdm.write("Starting training...")
    train(
        train_config,
        model,
        tokenizer,
        loss_fn,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        scaler,
        autocast_ctx,
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Llama-based model on a single GPU using LeCarnet dataset."
    )
    parser.add_argument(
        "--model_config",
        type=str,
        choices=["custom", "3M", "8M", "21M"],
        default="3M",
        help="Size of the model to train.",
    )
    parser.add_argument("--language", type=str, choices=["english", "french"], default="english",
                        help="Language of the dataset to train on.")
    args = parser.parse_args()
    main(args)
