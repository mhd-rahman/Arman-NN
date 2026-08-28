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
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            past_kv: cached (key, value) tensors of shape (b, heads, past_len, head_dim)
            attention_mask: optional (batch, seq_len) mask, 1 = keep, 0 = pad.
                            When provided, padded positions are masked out.

        Returns:
            output: (batch, seq_len, d_model)
            present_kv: updated (key, value) cache
        """
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (b, t, heads, head_dim)

        # Transpose to (b, heads, t, head_dim) for SDPA
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        is_decode_step = past_kv is not None
        if is_decode_step:
            past_k, past_v = past_kv
            k_t = torch.cat([past_k, k_t], dim=2)
            v_t = torch.cat([past_v, v_t], dim=2)

        present_kv = (k_t, v_t)
        kv_len = k_t.size(2)

        # Build attention mask if padding is provided
        attn_bias = None
        use_causal_flag = not is_decode_step  # single/chunk decode handled explicitly below

        if attention_mask is not None:
            # attention_mask: (b, kv_len) — expand to (b, 1, q_len, kv_len)
            # 1 = attend, 0 = mask. Convert to additive bias.
            pad_mask = attention_mask[:, None, None, :].to(q_t.dtype)  # (b,1,1,kv_len)
            attn_bias = (1.0 - pad_mask) * torch.finfo(q_t.dtype).min

            # Add causal mask on top (query can't see future keys)
            causal = torch.ones(t, kv_len, dtype=torch.bool, device=x.device).tril(
                diagonal=kv_len - t
            )
            causal_bias = torch.where(
                causal, 0.0, torch.finfo(q_t.dtype).min
            ).to(q_t.dtype)
            attn_bias = attn_bias + causal_bias[None, None, :, :]
            use_causal_flag = False

        # Flash Attention path (fp16/bf16, no explicit mask, training/prefill)
        if (
            self.use_flash
            and attn_bias is None
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            # flash_attn_func expects (b, seq_len, heads, head_dim)
            if is_decode_step:
                # Re-derive (b, t, heads, head_dim) layout for flash with cached kv
                q_f = q_t.transpose(1, 2)
                k_f = k_t.transpose(1, 2)
                v_f = v_t.transpose(1, 2)
                y = flash_attn_func(
                    q_f, k_f, v_f,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=True,
                )
            else:
                y = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=True,
                )
            y = y.contiguous().view(b, t, c)
        else:
            # PyTorch SDPA path — handles fp32, masks, and causal correctly
            y = torch.nn.functional.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=use_causal_flag,
            )
            y = y.transpose(1, 2).contiguous().view(b, t, c)

        return self.out(y), present_kv
