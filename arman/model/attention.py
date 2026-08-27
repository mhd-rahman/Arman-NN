import torch
from torch import nn


# Try to import Flash Attention 2 for optimized attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, use_flash: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.use_flash = use_flash and FLASH_ATTN_AVAILABLE

    def forward(
        self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        if past_kv is not None:
            # KV-cache path: q is (b, t, heads, head_dim), k/v need concat
            past_k, past_v = past_kv  # (b, heads, past_len, head_dim)
            k_t = k.transpose(1, 2)  # (b, heads, t, head_dim)
            v_t = v.transpose(1, 2)
            k_full = torch.cat([past_k, k_t], dim=2)
            v_full = torch.cat([past_v, v_t], dim=2)
            present_kv = (k_full, v_full)

            # Cached generation — use standard SDPA (flash doesn't help for single-token)
            q_t = q.transpose(1, 2)  # (b, heads, t, head_dim)
            y = torch.nn.functional.scaled_dot_product_attention(
                q_t, k_full, v_full, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )
            y = y.transpose(1, 2).contiguous().view(b, t, c)
        else:
            # Training / prefill path
            present_kv = (k.transpose(1, 2), v.transpose(1, 2))

            if self.use_flash:
                # flash_attn_func expects (b, seq_len, heads, head_dim)
                # q, k, v are already (b, t, heads, head_dim) from the view+unbind above
                y = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=True,
                )
                y = y.contiguous().view(b, t, c)
            else:
                # Fallback to PyTorch SDPA
                q_t = q.transpose(1, 2)
                k_t = k.transpose(1, 2)
                v_t = v.transpose(1, 2)
                y = torch.nn.functional.scaled_dot_product_attention(
                    q_t, k_t, v_t, attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0, is_causal=True
                )
                y = y.transpose(1, 2).contiguous().view(b, t, c)

        return self.out(y), present_kv
