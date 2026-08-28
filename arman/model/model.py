import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from .block import ArmanBlock
from .graph import GraphProcessor
from .memory import NeuralMemory


class ArmanNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([ArmanBlock(config) for _ in range(config.n_layers)])

        self.graph = GraphProcessor(config.d_model, config.graph_layers) if config.use_graph else None
        self.memory = NeuralMemory(config.d_model, config.memory_slots, config.memory_top_k) if config.use_memory else None

        n_global_paths = 1 + int(self.graph is not None) + int(self.memory is not None)
        self.global_router = nn.Linear(config.d_model, n_global_paths, bias=False) if config.use_router and n_global_paths > 1 else None
        self.final_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to reduce memory at the cost of ~30% slower training."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(self, input_ids, graph_nodes=None, adjacency=None, targets=None,
                past_key_values=None, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len) token ids
            graph_nodes: optional graph node features
            adjacency: optional adjacency matrix
            targets: optional target ids for loss computation
            past_key_values: list of (past_kv, past_ssm_state) per layer for cached generation
            attention_mask: optional (batch, seq_len) mask, 1 = keep, 0 = pad
        Returns:
            dict with logits, loss, aux_loss, memory_write_signal, present_key_values
        """
        b, t = input_ids.shape
        use_cache = past_key_values is not None

        # Compute position offset from cache length
        if use_cache and past_key_values[0][0] is not None:
            pos_offset = past_key_values[0][0][0].size(2)  # past_k shape: (b, heads, past_len, head_dim)
        else:
            pos_offset = 0

        if t + pos_offset > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {t + pos_offset} exceeds max_seq_len={self.config.max_seq_len}"
            )

        pos = torch.arange(pos_offset, pos_offset + t, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)[None, :, :]

        aux_loss = x.new_zeros(())
        present_key_values = []

        for i, block in enumerate(self.blocks):
            layer_past_kv = None
            layer_past_ssm = None
            if use_cache and past_key_values[i] is not None:
                layer_past_kv, layer_past_ssm = past_key_values[i]

            if self.gradient_checkpointing and self.training and not use_cache:
                x, block_aux, present_kv, present_ssm = gradient_checkpoint(
                    block, x, layer_past_kv, layer_past_ssm, attention_mask,
                    use_reentrant=False
                )
            else:
                x, block_aux, present_kv, present_ssm = block(
                    x, past_kv=layer_past_kv, past_ssm_state=layer_past_ssm,
                    attention_mask=attention_mask,
                )
            aux_loss = aux_loss + block_aux
            present_key_values.append((present_kv, present_ssm))

        paths = [x]
        if self.graph is not None:
            paths.append(self.graph(x, graph_nodes, adjacency))
        memory_write = None
        if self.memory is not None:
            memory_read, memory_write = self.memory(x)
            paths.append(memory_read)

        if len(paths) > 1:
            stacked = torch.stack(paths, dim=-2)
            if self.global_router is not None:
                weights = torch.softmax(self.global_router(x), dim=-1).unsqueeze(-1)
                x = (stacked * weights).sum(dim=-2)
            else:
                x = stacked.mean(dim=-2)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            # Mask out padded positions in the loss if attention_mask is provided
            if attention_mask is not None:
                targets = targets.masked_fill(attention_mask == 0, -100)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            loss = loss + 0.01 * aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss,
            "memory_write_signal": memory_write,
            "present_key_values": present_key_values,
        }

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
