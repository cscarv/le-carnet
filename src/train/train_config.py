from dataclasses import dataclass
import torch


@dataclass
class TrainConfig:
    dataset_name: str = "MaxLSB/LeCarnet"
    tokenizer_name: str = "lightonai/pagnol-small"
    output_dir: str = "checkpoints/"
    cache_dir: str = "cache/"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_steps: int = 500
    gradient_accumulation_steps: int = 2
    train_batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 5e-4
    num_warmup_steps: int = 500
    max_train_steps: int = 45000
    block_size: int = 512
    num_workers: int = 4
