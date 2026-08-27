from dataclasses import dataclass


@dataclass
class ArmanConfig:
    vocab_size: int = 32000
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    max_seq_len: int = 1024
    dropout: float = 0.0

    # SSM
    ssm_state_size: int = 64
    ssm_kernel_size: int = 5

    # Dense MLP
    mlp_hidden: int = 1024

    # MoE
    n_experts: int = 4
    moe_top_k: int = 2
    expert_hidden: int = 1024

    # Graph processor
    graph_layers: int = 2

    # Memory
    memory_slots: int = 64
    memory_top_k: int = 4

    # Feature switches
    use_attention: bool = True
    use_ssm: bool = True
    use_mlp: bool = True
    use_moe: bool = True
    use_graph: bool = True
    use_memory: bool = True
    use_router: bool = True

    tie_embeddings: bool = True
