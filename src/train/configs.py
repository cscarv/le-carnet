import torch
from dataclasses import dataclass


@dataclass
class TrainConfigEnglish:
    train_dataset_path: str = "/nobackup/users/scarv/multi-teacher-distillation/data/mixed_eng_fr_tinystories_1.0_max_4012794"
    val_dataset_path: str = "/nobackup/users/scarv/multi-teacher-distillation/data/tinystories_val"
    tokenizer_path: str = "/nobackup/users/scarv/multi-teacher-distillation/data/eng_fr_tokenizer/tokenizer.json"
    output_dir: str = "multi-teacher-distillation/le-carnet/checkpoints/english"
    load_checkpoint_path: str = "multi-teacher-distillation/le-carnet/checkpoints/english/checkpoint-epoch-0.pt"
    load_checkpoint: bool = False
    mixed_precision: bool = True
    cache_dir: str = "cache/"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_steps: int = 5000
    gradient_accumulation_steps: int = 2
    train_batch_size: int = 128
    eval_batch_size: int = 16
    learning_rate: float = 5e-4
    num_warmup_steps: int = 5000
    num_epochs: int = 10
    block_size: int = 512
    num_workers: int = 4

@dataclass
class TrainConfigFrench:
    train_dataset_path: str = "/nobackup/users/scarv/multi-teacher-distillation/data/mixed_eng_fr_tinystories_0.0_max_4012794"
    val_dataset_path: str = "/nobackup/users/scarv/multi-teacher-distillation/data/lecarnet_val"
    tokenizer_path: str = "/nobackup/users/scarv/multi-teacher-distillation/data/eng_fr_tokenizer/tokenizer.json"
    output_dir: str = "multi-teacher-distillation/le-carnet/checkpoints/french"
    load_checkpoint_path: str = "multi-teacher-distillation/le-carnet/checkpoints/french/checkpoint-epoch-0.pt"
    load_checkpoint: bool = False
    mixed_precision: bool = True
    cache_dir: str = "cache/"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_steps: int = 5000
    gradient_accumulation_steps: int = 2
    train_batch_size: int = 128
    eval_batch_size: int = 16
    learning_rate: float = 5e-4
    num_warmup_steps: int = 5000
    num_epochs: int = 10
    block_size: int = 512
    num_workers: int = 4


@dataclass
class CustomConfig:
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    block_size: int = 512
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True


@dataclass
class ModelConfig_3M:
    hidden_size: int = 64
    intermediate_size: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    block_size: int = 512
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True


@dataclass
class ModelConfig_8M:
    hidden_size: int = 128
    intermediate_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    block_size: int = 512
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True


@dataclass
class ModelConfig_21M:
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    block_size: int = 512
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True
