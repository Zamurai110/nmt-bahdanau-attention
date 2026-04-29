"""
Bahdanau (Additive) Attention Mechanism
----------------------------------------
Reference: Bahdanau et al., "Neural Machine Translation by Jointly Learning to
           Align and Translate" (ICLR 2015).  https://arxiv.org/abs/1409.0473

Score function:   e_ij = v^T · tanh( W_a·s_{i-1}  +  U_a·h_j )
Context vector:   c_i  = Σ_j  α_ij · h_j          (α = softmax(e))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Args:
        enc_hid_dim : encoder hidden size (× 2 if bidirectional)
        dec_hid_dim : decoder hidden size
    """

    def __init__(self, enc_hid_dim: int, dec_hid_dim: int):
        super().__init__()
        self.W_a = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
        self.U_a = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        self.v   = nn.Linear(dec_hid_dim, 1,           bias=False)

    def forward(self,
                decoder_hidden: torch.Tensor,
                encoder_outputs: torch.Tensor,
                src_mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden  : (batch, dec_hid_dim)      – previous decoder state
            encoder_outputs : (src_len, batch, enc_hid_dim)
            src_mask        : (src_len, batch) BoolTensor – True where <pad>

        Returns:
            context  : (batch, enc_hid_dim)   weighted sum of encoder outputs
            alpha    : (batch, src_len)        attention weights
        """
        src_len, batch, _ = encoder_outputs.shape

        # Expand decoder hidden → (src_len, batch, dec_hid_dim)
        dec_hidden_exp = decoder_hidden.unsqueeze(0).expand(src_len, -1, -1)

        # Energy scores   e : (src_len, batch, 1)
        energy = self.v(
            torch.tanh(self.W_a(encoder_outputs) + self.U_a(dec_hidden_exp))
        )
        energy = energy.squeeze(-1)          # (src_len, batch)

        # Mask padding positions with −∞ before softmax
        if src_mask is not None:
            energy = energy.masked_fill(src_mask, float("-1e9"))

        alpha   = F.softmax(energy, dim=0)   # (src_len, batch)

        # Context vector  c : (batch, enc_hid_dim)
        context = (alpha.unsqueeze(-1) * encoder_outputs).sum(0)

        return context, alpha.t()            # alpha returned as (batch, src_len)
