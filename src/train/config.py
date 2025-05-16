from dataclasses import dataclass


@dataclass
class ModelConfig_3M:
    vocab_size: int = 50258
    hidden_size: int = 64
    intermediate_size: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    block_size: int = 512
    max_position_embeddings: int = 2048
    pad_token_id: int = 50257
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True
