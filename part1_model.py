"""
Byte-level LSTM language model

This file was written with assistance from generative AI.
All AI-generated code was read over and checked for correctness.
"""

import torch
from torch import nn

from evaluator import PAD_ID, VOCAB_SIZE


class LSTMModel(nn.Module):
    """
    Unidirectional LSTM model over UTF-8 bytes.

    input: [batch_size, block_length]
    output: [batch_size, block_length, vocab_size]
    """

    def __init__(
        self, 
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout,
        recurrent_bias,
        output_bias,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=PAD_ID)
        self.embed_dropout = nn.Dropout(dropout)
        # LSTM.dropout covers every layer except the last, so the output dropout
        # sits on the final layer before the vocabulary projection.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bias=recurrent_bias,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, VOCAB_SIZE, bias=output_bias)

    @classmethod
    def from_config(cls, config):
        """Build the network from the model section of the YAML config."""
        model_cfg = config["model"]
        return cls(
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            recurrent_bias=model_cfg["recurrent_bias"],
            output_bias=model_cfg["output_bias"],
        )

    def forward(self, input_ids):
        """Return logits at each position. Hidden state starts at zero."""
        hidden = self.embedding(input_ids)
        hidden = self.embed_dropout(hidden)
        outputs, _ = self.lstm(hidden)
        outputs = self.output_dropout(outputs)
        return self.output(outputs)


def load_checkpoint(path, device):
    """Rebuild the LSTM from a best-dev checkpoint and load its weights."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LSTMModel.from_config({"model": checkpoint["model"]})
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
