from dataclasses import dataclass


@dataclass
class ModelConfig_2M:
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
class ModelConfig_16M:
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


@dataclass
class ModelConfig_33M:
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 4
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
class ModelConfig_50M:
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
