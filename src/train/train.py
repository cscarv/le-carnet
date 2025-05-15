from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    get_scheduler,
)
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim import AdamW
from train.config import ModelConfig
import torch
from torch.utils.data import DataLoader

# https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section6_pt.ipynb#scrollTo=ziBdsf0IZO0I


class Tokenizer:
    """
    Tokenizer class to load and prepare the tokenizer
    """

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "openai-community/gpt2", hf_token="..."
        )
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )  # Not sure if we should do this

    def ready_tokenizer(self):
        return self.tokenizer


def get_dataset(dataset_name):
    """
    Load the dataset from Hugging Face Hub
    """
    train_dataset = load_dataset(
        dataset_name,
        split="train",
        cache_dir="./cache/",
    )
    val_dataset = load_dataset(
        dataset_name,
        split=f"validation",
        cache_dir="./cache/",
    )
    return train_dataset, val_dataset


def collate_fn(batch):
    texts = [item["text"] for item in batch]

    input_encodings = tokenizer(
        texts,
        max_length=ModelConfig.block_size,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_encodings["labels"] = input_encodings["input_ids"].clone()

    input_encodings["labels"][:, :-1] = input_encodings["input_ids"][:, 1:]
    input_encodings["labels"][:, -1] = tokenizer.eos_token_id
    # Add BOS token ?

    return input_encodings


def get_dataloader(
    split,
    dataset,
    batch_size,
):
    is_train = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=is_train,
    )


def get_num_params(model):
    """
    Get the number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, eval_dataloader, accelerator, loss_fn):
    """
    Evaluate the model on the validation set
    """
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            input_batch = batch["input_ids"]
            target_batch = batch["labels"]
            outputs = calc_loss_batch(input_batch, target_batch, model, loss_fn)
            losses.append(outputs.item())

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


def calc_loss_batch(input_batch, target_batch, model, loss_fn):
    """Calculate loss for a single batch."""
    logits = model(input_batch)
    loss = loss_fn(logits.view(-1, logits.size(-1)), target_batch.view(-1))
    return loss


def train(
    model,
    loss_fn,
    num_train_epochs,
    train_dataloader,
    eval_dataloader,
    accelerator,
    optimizer,
    lr_scheduler,
    output_dir="./results",
):
    gradient_accumulation_steps = 8
    eval_steps = 5_000
    num_training_steps = 5000

    model.train()
    completed_steps = 0
    for epoch in range(num_train_epochs):
        for step, batch in tqdm(
            enumerate(train_dataloader, start=1), total=num_training_steps
        ):
            input_batch = batch["input_ids"]
            target_batch = batch["labels"]
            loss = calc_loss_batch(input_batch, target_batch, model, loss_fn)

            if step % 100 == 0:
                accelerator.print(
                    {
                        "steps": completed_steps,
                        "loss/train": loss.item() * gradient_accumulation_steps,
                    }
                )
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            if (step % (eval_steps * gradient_accumulation_steps)) == 0:
                eval_loss, perplexity = evaluate(
                    model, eval_dataloader, accelerator, loss_fn
                )
                accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
                model.train()
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    output_dir, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)
                    # repo.push_to_hub(
                    #     commit_message=f"Training in progress step {step}",
                    #     blocking=False,
                    # )


def get_llama_config(config):
    return LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_decoder_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.hidden_size * 4,
        max_position_embeddings=config.block_size,
        layer_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        attention_dropout=config.attention_dropout,
        dropout=config.dropout,
        device_map="auto",
        gradient_checkpointing=True,
    )


def main():
    # Load the dataset
    dataset_name = "MaxLSB/French-TinyStories"  # Change the name
    tokenizer_name = "gpt2"  # Maybe use a french tokenizer ?
    num_train_epochs = 1
    num_update_steps_per_epoch = 10**6 / 32
    num_training_steps = 10**6 / 32

    # Get the model config
    config = ModelConfig()

    # Load the datasets
    ds_train, ds_valid = get_dataset(dataset_name)
    # Load the tokenizer
    global tokenizer
    tokenizer = Tokenizer().ready_tokenizer()

    # Initialize the model with the llama config
    llama_config = get_llama_config(config)
    model = LlamaForCausalLM(llama_config)

    # Print the number of parameters
    print(f"Model has {get_num_params(model)} parameters")

    # Instanciate the dataloaders
    train_dataloader = get_dataloader(
        "train",
        ds_train,
        batch_size=32,
    )
    eval_dataloader = get_dataloader(
        "validation",
        ds_valid,
        batch_size=32,
    )
    # Instantiate the accelerator
    accelerator = Accelerator(fp16=True)

    # Instanciate the loss and the optimizer
    loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
    optimizer = AdamW(model.parameters(), lr=5e-4)

    # Prepare the model, optimizer and dataloaders
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=1_000,
        num_training_steps=num_training_steps,
    )

    # Train the model
    train(
        model,
        loss_fn,
        num_train_epochs,
        train_dataloader,
        eval_dataloader,
        accelerator,
        optimizer,
        lr_scheduler,
    )


if __name__ == "__main__":
    main()
