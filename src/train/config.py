from dataclasses import dataclass
import torch

# To be changed


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    block_size: int = 512
    hidden_size: int = 512
    num_decoder_layers: int = 12
    num_attention_heads: int = 8
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_dropout: float = 0.1
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-6
    d_c_kv: int = 128
    d_c_q: int = 128
    d_rotate: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    total_iters: int = 10000
    eval_iters: int = 50
    effective_batch_size: int = 512
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    gradient_accumulation_steps: int = effective_batch_size // batch_size
    max_grad_norm: float = 1.0
    num_workers = 8
    dtype: str = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
