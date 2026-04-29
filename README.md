# Neural Machine Translation with Bahdanau Attention

> English → French translation using a Seq2Seq encoder-decoder with Bahdanau (additive) attention, teacher forcing, and beam search decoding.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Results

| Decoding  | BLEU Score | Dataset |
|-----------|-----------|---------|
| Greedy    | —         | —       |
| Beam (k=5)| **50.31** | ~175k pairs |

> Fill in after running `python evaluate.py`

---

## Architecture

```
Input (EN)
   │
   ▼
┌──────────────────────────────────────┐
│  Embedding  →  Bi-GRU Encoder       │  2 layers, 512 hidden, dropout 0.3
│  h1  h2  h3 … hN  (all states)      │
└──────────────────────────────────────┘
        │  (all encoder hidden states)
        ▼
┌──────────────────────────────────────────────────────┐
│  Bahdanau Attention                                  │
│  e_ij = vᵀ · tanh(W_a·s_{i-1}  +  U_a·h_j)         │
│  α_ij = softmax(e_ij)                                │
│  c_i  = Σ α_ij · h_j                                │
└──────────────────────────────────────────────────────┘
        │  context vector c_i
        ▼
┌──────────────────────────────────────┐
│  GRU Decoder  →  Linear  →  Softmax │  Teacher forcing (p=0.5)
└──────────────────────────────────────┘
        │
        ▼
  Output (FR)  via  Beam Search (k=5)
```

### Key design choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Encoder RNN | Bidirectional GRU | Captures left & right context for each source word |
| Attention | Bahdanau additive | Learnable alignment, no dimensionality constraint |
| Teacher Forcing | p = 0.5 | Balances fast convergence vs exposure bias |
| Decoding | Beam search (k=5) | Higher BLEU than greedy, manageable compute |
| Length penalty | α = 0.75 | Prevents beam search from favouring short outputs |

---

## Project Structure

```
nmt-bahdanau-attention/
├── config.py               ← all hyperparameters in one place
├── train.py                ← training loop (teacher forcing, early stop)
├── evaluate.py             ← BLEU evaluation on test split
├── translate.py            ← interactive / CLI inference
├── requirements.txt
├── data/
│   └── dataset.py          ← download, tokenise, build vocab, DataLoaders
├── model/
│   ├── attention.py        ← BahdanauAttention module
│   ├── encoder.py          ← Bidirectional GRU encoder
│   ├── decoder.py          ← Single-step attention decoder
│   └── seq2seq.py          ← Seq2Seq wrapper (teacher forcing)
└── utils/
    └── beam_search.py      ← length-normalised beam search
```

---

## Quick Start

### 1 · Clone and install

```bash
git clone https://github.com/<YOUR_USERNAME>/nmt-bahdanau-attention.git
cd nmt-bahdanau-attention
pip install -r requirements.txt
```

### 2 · Train

```bash
python train.py
```

Training auto-downloads the Anki English–French dataset (~175 k pairs), filters sentences to `MAX_LEN=20` tokens, and saves the best checkpoint to `checkpoints/best_model.pt`.

### 3 · Evaluate BLEU

```bash
python evaluate.py           # beam search (default)
python evaluate.py --greedy  # greedy decoding
```

### 4 · Translate interactively

```bash
# Single sentence
python translate.py --sentence "she speaks french very well ."

# Interactive REPL
python translate.py
```

### 5 · Monitor with TensorBoard

```bash
tensorboard --logdir logs/
```

---

## Configuration

All hyperparameters live in `config.py`. Key knobs:

```python
MAX_LEN        = 20       # filter out longer sentence pairs
NUM_PAIRS      = None     # cap dataset size (set e.g. 50_000 for fast debug)
EMB_DIM        = 256
HID_DIM        = 512
NUM_EPOCHS     = 30
LR             = 3e-4
TEACHER_FORCING= 0.5
BEAM_SIZE      = 5
BEAM_ALPHA     = 0.75
```

---

## How Teacher Forcing Works

```
Step t:  with probability p  → feed ground-truth  y_{t-1}   (teacher forcing)
         with probability 1-p → feed model's own   ŷ_{t-1}   (free running)
```

High `p` → fast convergence; low `p` → better robustness at inference. A value of 0.5 strikes a balance used widely in practice.

---

## How Beam Search Works

Instead of greedily picking `argmax` at each step, beam search maintains the **top-k partial hypotheses** and scores them jointly with a length penalty:

```
lp(Y) = ((5 + |Y|) / 6) ^ α
score = log P(Y|X) / lp(Y)
```

This yields higher BLEU scores than greedy decoding at the cost of `O(k)` more decoder forward passes.

---

## References

1. Bahdanau, D., Cho, K., Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)  
2. Sutskever, I., Vinyals, O., Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks.* NeurIPS 2014. [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)  
3. Wu, Y. et al. (2016). *Google's Neural Machine Translation System.* [arXiv:1609.08144](https://arxiv.org/abs/1609.08144)  

---

## License

MIT
