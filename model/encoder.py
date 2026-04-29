"""
Bi-directional GRU Encoder
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Args:
        vocab_size : source vocabulary size
        emb_dim    : embedding dimension
        hid_dim    : hidden state size *per direction*
        n_layers   : number of GRU layers
        dropout    : dropout probability
    """

    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(
            emb_dim, hid_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=False,
        )
        self.dropout   = nn.Dropout(dropout)
        # Project concatenated bi-GRU hidden to dec_hid_dim (= hid_dim)
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src : (src_len, batch)  – token indices

        Returns:
            outputs : (src_len, batch, hid_dim*2)  – all encoder hidden states
            hidden  : (batch, hid_dim)             – decoder-ready initial state
        """
        embedded = self.dropout(self.embedding(src))   # (src_len, B, emb)
        outputs, hidden = self.rnn(embedded)
        # hidden : (n_layers*2, B, hid_dim)  – take last layer both directions
        # Combine forward and backward last-layer hidden states
        hidden_fwd = hidden[-2]   # (B, hid)
        hidden_bwd = hidden[-1]   # (B, hid)
        hidden = torch.tanh(self.fc_hidden(
            torch.cat([hidden_fwd, hidden_bwd], dim=-1)
        ))                        # (B, hid)
        return outputs, hidden
