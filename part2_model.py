"""
Byte-level decoder-only Transformer language model.
"""

import torch
from torch import nn

from evaluator import BLOCK_LENGTH, PAD_ID, VOCAB_SIZE


class TransformerLanguageModel(nn.Module):
    """Causal Transformer language model over UTF-8 bytes.

    input shape: [batch size, block length]
    output shape: [batch size, block length, vocab size]
    """

    def __init__(
        self,
        hidden_dim,
        num_layers,
        num_heads,
        feedforward_dim,
        dropout,
        activation,
        norm_first,
        output_bias,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embedding = nn.Embedding(VOCAB_SIZE, hidden_dim, padding_idx=PAD_ID)
        self.position = nn.Embedding(BLOCK_LENGTH, hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        # Pre-norm layers normalize their inputs, so the stack needs a final norm.
        final_norm = nn.LayerNorm(hidden_dim) if norm_first else None
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=final_norm,
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(hidden_dim, VOCAB_SIZE, bias=output_bias)

    @classmethod
    def from_config(cls, config):
        """Build the network from the model section of the YAML config."""
        model_cfg = config["model"]
        return cls(
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg["num_heads"],
            feedforward_dim=model_cfg["feedforward_dim"],
            dropout=model_cfg["dropout"],
            activation=model_cfg["activation"],
            norm_first=model_cfg["norm_first"],
            output_bias=model_cfg["output_bias"],
        )

    def forward(self, input_ids):
        """Return logits at each position. Position ids start at zero."""
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.embedding(input_ids) + self.position(positions)
        hidden = self.embed_dropout(hidden)
        # True entries are blocked, so position t attends only to 0..t.
        causal = torch.ones(length, length, dtype=torch.bool, device=input_ids.device)
        causal = causal.triu(diagonal=1)
        return self.output(self.encoder(hidden, mask=causal))


def load_checkpoint(path, device):
    """Rebuild the Transformer from a best-dev checkpoint and load its weights."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TransformerLanguageModel.from_config({"model": checkpoint["model"]})
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
