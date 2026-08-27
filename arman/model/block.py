import torch
from torch import nn
from .attention import CausalSelfAttention
from .ssm import SelectiveSSM
from .mlp import SwiGLU
from .moe import SparseMoE
from .router import PathRouter


class ArmanBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.config = config
        self.norm1 = nn.RMSNorm(d)
        self.norm2 = nn.RMSNorm(d)

        self.attn = CausalSelfAttention(d, config.n_heads, config.dropout) if config.use_attention else None
        self.ssm = SelectiveSSM(d, config.ssm_state_size) if config.use_ssm else None
        self.fusion_gate = nn.Linear(2 * d, d) if self.attn is not None and self.ssm is not None else None

        self.mlp = SwiGLU(d, config.mlp_hidden) if config.use_mlp else None
        self.moe = SparseMoE(d, config.expert_hidden, config.n_experts, config.moe_top_k) if config.use_moe else None
        n_paths = int(self.mlp is not None) + int(self.moe is not None)
        self.router = PathRouter(d, n_paths) if config.use_router and n_paths > 1 else None

    def _sequence_mix(self, h, past_kv=None, past_ssm_state=None):
        present_kv = None
        present_ssm_state = None

        a = None
        s = None

        if self.attn is not None:
            a, present_kv = self.attn(h, past_kv=past_kv)
        if self.ssm is not None:
            s, present_ssm_state = self.ssm(h, past_state=past_ssm_state)

        if a is not None and s is not None:
            gate = torch.sigmoid(self.fusion_gate(torch.cat([a, s], dim=-1)))
            out = gate * a + (1.0 - gate) * s
        else:
            out = a if a is not None else s

        return out, present_kv, present_ssm_state

    def _ff_mix(self, h):
        paths = []
        aux = h.new_zeros(())
        if self.mlp is not None:
            paths.append(self.mlp(h))
        if self.moe is not None:
            moe_out, moe_aux = self.moe(h)
            paths.append(moe_out)
            aux = aux + moe_aux
        if not paths:
            return torch.zeros_like(h), aux
        if len(paths) == 1:
            return paths[0], aux
        weights = self.router(h)
        stacked = torch.stack(paths, dim=-2)
        return (stacked * weights.unsqueeze(-1)).sum(dim=-2), aux

    def forward(self, x, past_kv=None, past_ssm_state=None):
        seq, present_kv, present_ssm_state = self._sequence_mix(
            self.norm1(x), past_kv=past_kv, past_ssm_state=past_ssm_state
        )
        if seq is not None:
            x = x + seq
        ff, aux = self._ff_mix(self.norm2(x))
        x = x + ff
        return x, aux, present_kv, present_ssm_state
