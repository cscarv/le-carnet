from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    LlamaConfig,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim import AdamW
import torch

# https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section6_pt.ipynb#scrollTo=ziBdsf0IZO0I


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
        )  # Not sure if this is correct

    def ready_tokenizer(self):
        return self.tokenizer


def tokenize(element, tokenizer, context_length):
    """
    Tokenize the input text
    """
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


def get_num_params(model):
    """
    Get the number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, eval_dataloader, accelerator):
    """
    Evaluate the model on the validation set
    """
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


def train():
    gradient_accumulation_steps = 8
    eval_steps = 5_000

    model.train()
    completed_steps = 0
    for epoch in range(num_train_epochs):
        for step, batch in tqdm(
            enumerate(train_dataloader, start=1), total=num_training_steps
        ):
            logits = model(batch["input_ids"]).logits
            loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)
            if step % 100 == 0:
                accelerator.print(
                    {
                        "samples": step * samples_per_step,
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
                eval_loss, perplexity = evaluate()
                accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
                model.train()
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    output_dir, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)
                    repo.push_to_hub(
                        commit_message=f"Training in progress step {step}",
                        blocking=False,
                    )


def main():
    # Load the dataset
    dataset_name = "my_dataset_name"  # Replace with your dataset name
    tokenizer_name = "gpt2"  # Maybe use a french tokenizer ?
    ds_train, ds_valid = get_dataset(dataset_name)

    tokenizer = Tokenizer().ready_tokenizer()
    config = LlamaConfig.from_json_file("./config.json")

    # Initialize the model with the llama config
    model = LlamaForCausalLM(config)

    # Print the number of parameters
    print(f"Model has {get_num_params(model)} parameters")

    optimizer = AdamW(get_grouped_params(model), lr=5e-4)
    accelerator = Accelerator(fp16=True)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_train_epochs = 1
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=1_000,
        num_training_steps=num_training_steps,
    )


if __name__ == "__main__":
    main()
