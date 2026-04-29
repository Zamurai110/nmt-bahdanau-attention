"""
Evaluate the trained model on the test split.
Computes corpus-level BLEU score using sacrebleu.
"""

import pickle, torch
from pathlib import Path

from sacrebleu.metrics import BLEU

import config as C
from data.dataset   import get_data_loaders, PAD_IDX, EOS_IDX
from model.encoder  import Encoder
from model.decoder  import Decoder
from model.seq2seq  import Seq2Seq
from utils.beam_search import beam_search_decode


def load_model(src_vocab_size: int, tgt_vocab_size: int) -> Seq2Seq:
    enc = Encoder(src_vocab_size, C.EMB_DIM, C.HID_DIM,
                  C.ENC_LAYERS, C.ENC_DROPOUT)
    dec = Decoder(tgt_vocab_size, C.EMB_DIM, C.HID_DIM * 2,
                  C.HID_DIM, C.DEC_DROPOUT)
    model = Seq2Seq(enc, dec, C.DEVICE).to(C.DEVICE)
    ckpt  = torch.load(f"{C.CHECKPOINTS}/best_model.pt",
                       map_location=C.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def greedy_decode(model, src, tgt_vocab, max_len=C.MAX_LEN + 10):
    """Fastest decoding – argmax at each step."""
    src_mask     = (src == PAD_IDX)
    enc_out, hid = model.encoder(src)

    token = torch.tensor([1], device=C.DEVICE)   # <sos>
    tokens = []
    for _ in range(max_len):
        pred, hid, _ = model.decoder(token, hid, enc_out, src_mask)
        token = pred.argmax(-1)
        if token.item() == EOS_IDX:
            break
        tokens.append(token.item())
    return tgt_vocab.decode(tokens)


def evaluate(use_beam: bool = True, n_examples: int = 10):
    with open(f"{C.CHECKPOINTS}/src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open(f"{C.CHECKPOINTS}/tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)

    model = load_model(len(src_vocab), len(tgt_vocab))

    _, _, test_loader, _, _ = get_data_loaders()

    hypotheses = []
    references = []

    for src_batch, trg_batch in test_loader:
        for i in range(src_batch.size(1)):
            src = src_batch[:, i:i+1].to(C.DEVICE)
            ref = trg_batch[:, i].tolist()
            ref_str = tgt_vocab.decode(
                [t for t in ref if t not in {0, 1, 2, 3}]
            )
            if use_beam:
                hyps, _ = beam_search_decode(model, src, src_vocab, tgt_vocab)
                hyp_str = hyps[0] if hyps else ""
            else:
                hyp_str = greedy_decode(model, src, tgt_vocab)

            hypotheses.append(hyp_str)
            references.append(ref_str)

    bleu = BLEU()
    result = bleu.corpus_score(hypotheses, [references])

    mode = "Beam Search" if use_beam else "Greedy"
    print(f"\n{'='*50}")
    print(f"  Decoding : {mode}")
    print(f"  Corpus BLEU: {result.score:.2f}")
    print(f"{'='*50}\n")

    # Print a few examples
    print("Sample translations:")
    for i in range(min(n_examples, len(hypotheses))):
        print(f"  REF : {references[i]}")
        print(f"  HYP : {hypotheses[i]}")
        print()

    return result.score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding instead of beam search")
    args = parser.parse_args()
    evaluate(use_beam=not args.greedy)
