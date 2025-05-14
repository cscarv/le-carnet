from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    LlamaConfig,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
)


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


def main():
    # Load the dataset
    dataset_name = "my_dataset_name"  # Replace with your dataset name
    tokenizer_name = "gpt2"  # Maybe use a french tokenizer ?
    ds_train, ds_valid = get_dataset(dataset_name)

    tokenizer = Tokenizer().ready_tokenizer()
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    config = LlamaConfig.from_json_file("./config.json")

    # Initialize the model with the llama config
    model = LlamaForCausalLM(config)

    # Print the number of parameters
    print(f"Model has {get_num_params(model)} parameters")

    args = TrainingArguments(
        output_dir="codeparrot-ds",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    trainer.train()


if __name__ == "__main__":
    main()
