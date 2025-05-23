from dataclasses import dataclass
import torch


@dataclass
class TrainConfig:
    dataset_name: str = "MaxLSB/LeCarnet"
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
    output_dir: str = "checkpoints/"
    cache_dir: str = "cache/"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_steps: int = 500
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 8
    train_batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 5e-4
    num_warmup_steps: int = 200
    max_train_steps: int = 10000
    block_size: int = 512
