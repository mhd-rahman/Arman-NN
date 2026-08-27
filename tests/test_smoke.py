import torch
from arman.model import ArmanConfig, ArmanNN


def test_forward_and_backward():
    cfg = ArmanConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=4, max_seq_len=16,
                      mlp_hidden=128, expert_hidden=128, n_experts=4, moe_top_k=2,
                      ssm_state_size=16, memory_slots=8, graph_layers=1)
    model = ArmanNN(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    nodes = torch.randn(2, 5, cfg.d_model)
    adj = torch.eye(5).repeat(2, 1, 1)
    out = model(x, graph_nodes=nodes, adjacency=adj, targets=y)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert out["loss"] is not None
    out["loss"].backward()
