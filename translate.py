"""
Interactive translation CLI.
Usage:
    python translate.py --sentence "she speaks french very well ."
    python translate.py          # interactive REPL
"""

import argparse, pickle, torch

import config as C
from data.dataset   import tokenise, SOS_IDX, PAD_IDX
from model.encoder  import Encoder
from model.decoder  import Decoder
from model.seq2seq  import Seq2Seq
from utils.beam_search import beam_search_decode


def load_model_and_vocabs():
    with open(f"{C.CHECKPOINTS}/src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open(f"{C.CHECKPOINTS}/tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)

    enc = Encoder(len(src_vocab), C.EMB_DIM, C.HID_DIM,
                  C.ENC_LAYERS, C.ENC_DROPOUT)
    dec = Decoder(len(tgt_vocab), C.EMB_DIM, C.HID_DIM * 2,
                  C.HID_DIM, C.DEC_DROPOUT)
    model = Seq2Seq(enc, dec, C.DEVICE).to(C.DEVICE)
    ckpt  = torch.load(f"{C.CHECKPOINTS}/best_model.pt",
                       map_location=C.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, src_vocab, tgt_vocab


def translate(sentence: str, model, src_vocab, tgt_vocab) -> str:
    tokens = tokenise(sentence)
    ids    = [SOS_IDX] + src_vocab.encode(tokens) + [2]  # 2 = <eos>
    src    = torch.tensor(ids, device=C.DEVICE).unsqueeze(1)  # (len, 1)
    translations, scores = beam_search_decode(model, src, src_vocab, tgt_vocab)
    return translations[0] if translations else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", "-s", type=str, default=None,
                        help="Sentence to translate (English)")
    args   = parser.parse_args()

    print("Loading model…")
    model, sv, tv = load_model_and_vocabs()
    print("Model ready.\n")

    if args.sentence:
        result = translate(args.sentence, model, sv, tv)
        print(f"EN → {C.TGT_LANG.upper()}")
        print(f"  Input  : {args.sentence}")
        print(f"  Output : {result}")
    else:
        print(f"Interactive EN → {C.TGT_LANG.upper()} translator  (type 'quit' to exit)\n")
        while True:
            try:
                src = input("EN > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if src.lower() in {"quit", "exit", "q"}:
                break
            if src:
                print(f"FR > {translate(src, model, sv, tv)}\n")


if __name__ == "__main__":
    main()
