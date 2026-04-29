"""
Seq2Seq wrapper: Encoder + Decoder + Teacher Forcing
"""

import random
import torch
import torch.nn as nn

import config as C
from data.dataset import SOS_IDX, PAD_IDX


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device: str):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    # ------------------------------------------------------------------
    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                teacher_forcing_ratio: float = C.TEACHER_FORCING
                ) -> torch.Tensor:
        """
        Args:
            src : (src_len, batch)
            trg : (trg_len, batch)
            teacher_forcing_ratio : float in [0, 1]

        Returns:
            outputs : (trg_len - 1, batch, tgt_vocab)
        """
        trg_len, batch = trg.shape
        tgt_vocab_size  = self.decoder.fc_out.out_features

        # Padding mask: True where src == <pad>
        src_mask = (src == PAD_IDX)   # (src_len, batch)

        enc_outputs, hidden = self.encoder(src)

        # Tensor to collect decoder predictions
        outputs = torch.zeros(trg_len - 1, batch, tgt_vocab_size,
                              device=self.device)

        input_token = trg[0]   # <sos> token for every sentence in batch

        for t in range(1, trg_len):
            pred, hidden, _ = self.decoder(
                input_token, hidden, enc_outputs, src_mask
            )
            outputs[t - 1] = pred

            # Teacher forcing
            use_teacher = random.random() < teacher_forcing_ratio
            input_token  = trg[t] if use_teacher else pred.argmax(-1)

        return outputs

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
