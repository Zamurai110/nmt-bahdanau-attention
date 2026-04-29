"""
Beam Search Decoder
--------------------
Implements length-normalised beam search with configurable beam width
and alpha penalty (Wu et al., 2016).

Length penalty:  lp(Y) = ((5 + |Y|) / 6) ^ alpha
"""

import torch
import torch.nn.functional as F

import config as C
from data.dataset import SOS_IDX, EOS_IDX, PAD_IDX


@torch.no_grad()
def beam_search_decode(
    model,
    src: torch.Tensor,
    src_vocab,
    tgt_vocab,
    beam_size: int = C.BEAM_SIZE,
    max_len:   int = C.MAX_LEN + 10,
    alpha:     float = C.BEAM_ALPHA,
) -> tuple[list[str], list[float]]:
    """
    Translate a single source sentence with beam search.

    Args:
        model     : trained Seq2Seq model (eval mode)
        src       : (src_len, 1)  – token indices (batch size = 1)
        src_vocab : source Vocab object (for reference only)
        tgt_vocab : target Vocab object
        beam_size : number of beams
        max_len   : maximum translation length
        alpha     : length-penalty exponent

    Returns:
        translations : list of decoded strings (best → worst)
        scores       : corresponding normalised log-probability scores
    """
    device = next(model.parameters()).device
    src    = src.to(device)

    src_mask     = (src == PAD_IDX)                  # (src_len, 1)
    enc_out, hid = model.encoder(src)
    # Expand to beam_size
    enc_out = enc_out.repeat(1, beam_size, 1)         # (src_len, B, enc_hid)
    src_mask= src_mask.repeat(1, beam_size)           # (src_len, B)
    hid     = hid.repeat(beam_size, 1)                # (B, dec_hid)

    # Beam state: list of (token_sequence, cumulative_log_prob, hidden)
    # Initialised with <sos>
    beams = [(
        [SOS_IDX],          # sequence so far
        0.0,                # cumulative log-prob
        hid[i].unsqueeze(0) # hidden state  (1, dec_hid)
    ) for i in range(1)]    # start with 1 beam, expand on first step

    completed = []

    for step in range(max_len):
        candidates = []

        for seq, score, h in beams:
            if seq[-1] == EOS_IDX:
                completed.append((seq, score))
                continue

            token = torch.tensor([seq[-1]], device=device)  # (1,)
            # Use only this beam's slice of encoder outputs
            beam_idx = len(candidates) % beam_size
            e_out    = enc_out[:, beam_idx:beam_idx+1, :]
            m_out    = src_mask[:, beam_idx:beam_idx+1]

            pred, h_new, _ = model.decoder(token, h.squeeze(0), e_out, m_out)
            # pred: (1, vocab_size)
            log_probs = F.log_softmax(pred, dim=-1).squeeze(0)  # (vocab,)

            topk_lp, topk_idx = log_probs.topk(beam_size)
            for lp, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
                candidates.append((seq + [idx], score + lp, h_new.unsqueeze(0)))

        if not candidates:
            break

        # Length-normalised score
        def norm_score(candidate):
            seq, sc, _ = candidate
            lp = ((5 + len(seq)) / 6) ** alpha
            return sc / lp

        candidates.sort(key=norm_score, reverse=True)
        beams = candidates[:beam_size]

        # Early stop if all beams are finished
        if all(b[0][-1] == EOS_IDX for b in beams):
            completed.extend([(b[0], b[1]) for b in beams])
            break

    if not completed:
        completed = [(b[0], b[1]) for b in beams]

    # Sort completed by normalised score
    completed.sort(key=lambda x: x[1] / (((5 + len(x[0])) / 6) ** alpha),
                   reverse=True)

    translations, scores = [], []
    for seq, sc in completed[:beam_size]:
        tokens = [i for i in seq if i not in {SOS_IDX, EOS_IDX, PAD_IDX}]
        translations.append(tgt_vocab.decode(tokens))
        norm = sc / (((5 + len(seq)) / 6) ** alpha)
        scores.append(round(norm, 4))

    return translations, scores
