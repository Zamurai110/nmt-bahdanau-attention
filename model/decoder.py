"""
Attention Decoder (single GRU step)
"""

import torch
import torch.nn as nn

from model.attention import BahdanauAttention


class Decoder(nn.Module):
    """
    One-step decoder that consumes the previous token, the previous hidden
    state, and an attention-weighted context vector to emit a distribution
    over the target vocabulary.

    Args:
        vocab_size  : target vocabulary size
        emb_dim     : embedding dimension
        enc_hid_dim : encoder hidden size (× 2 for bi-GRU)
        dec_hid_dim : decoder hidden size
        dropout     : dropout probability
    """

    def __init__(self, vocab_size: int, emb_dim: int,
                 enc_hid_dim: int, dec_hid_dim: int, dropout: float):
        super().__init__()
        self.attention = BahdanauAttention(enc_hid_dim, dec_hid_dim)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(
            emb_dim + enc_hid_dim,   # input = embed || context
            dec_hid_dim,
            batch_first=False,
        )
        self.fc_out  = nn.Linear(dec_hid_dim + enc_hid_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                trg_token: torch.Tensor,
                hidden: torch.Tensor,
                enc_outputs: torch.Tensor,
                src_mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            trg_token   : (batch,)            – previous target token
            hidden      : (batch, dec_hid)    – previous decoder hidden state
            enc_outputs : (src_len, batch, enc_hid*2)
            src_mask    : (src_len, batch)

        Returns:
            prediction  : (batch, vocab_size)  – log-softmax logits
            hidden      : (batch, dec_hid)     – updated hidden state
            alpha       : (batch, src_len)     – attention weights
        """
        trg_token = trg_token.unsqueeze(0)                    # (1, B)
        embedded  = self.dropout(self.embedding(trg_token))   # (1, B, emb)

        context, alpha = self.attention(hidden, enc_outputs, src_mask)
        # context: (B, enc_hid*2)

        rnn_input = torch.cat([embedded, context.unsqueeze(0)], dim=-1)
        # rnn_input: (1, B, emb + enc_hid*2)

        hidden_for_rnn = hidden.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        output, hidden = self.rnn(rnn_input, hidden_for_rnn)
        # output : (1, B, dec_hid)
        # hidden : (1, B, dec_hid)
        hidden = hidden.squeeze(0)     # (B, dec_hid)
        output = output.squeeze(0)     # (B, dec_hid)

        # Combine decoder output, context, and embedding for prediction
        prediction = self.fc_out(
            torch.cat([output, context, embedded.squeeze(0)], dim=-1)
        )                              # (B, vocab_size)

        return prediction, hidden, alpha
