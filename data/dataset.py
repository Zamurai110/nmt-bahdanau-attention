"""
Download, clean, tokenise, and build vocabularies for sentence-pair data.
"""

import os, re, random, zipfile, unicodedata
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import config as C

# ── Special tokens ─────────────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


# ── Vocabulary ─────────────────────────────────────────────────────────────
class Vocab:
    def __init__(self):
        self.word2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build(self, sentences, min_freq=2):
        counter = Counter(w for sent in sentences for w in sent)
        for word, freq in sorted(counter.items()):
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        return self

    def encode(self, tokens):
        return [self.word2idx.get(t, UNK_IDX) for t in tokens]

    def decode(self, indices, skip_special=True):
        special = {PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX}
        words = [
            self.idx2word.get(i, UNK_TOKEN)
            for i in indices
            if not skip_special or i not in special
        ]
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


# ── Text helpers ───────────────────────────────────────────────────────────
def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def normalise(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-z.!?]+", " ", s)
    return s.strip()

def tokenise(s):
    return normalise(s).split()


# ── Download / load raw pairs ──────────────────────────────────────────────
def download_data():
    """Download Anki tab-separated sentence pairs and return path to .txt file."""
    import urllib.request

    raw_dir = Path(C.DATA_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "pairs.zip"

    # 1. Download
    if not zip_path.exists():
        print(f"Downloading data from {C.DATA_URL} ...")
        req = urllib.request.Request(
            C.DATA_URL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req) as response:
            data = response.read()
        with open(zip_path, "wb") as f:
            f.write(data)
        print(f"  Saved {zip_path} ({len(data) // 1024} KB)")

    # 2. Inspect & extract
    with zipfile.ZipFile(zip_path) as zf:
        all_names = zf.namelist()
        print(f"  Files in zip: {all_names}")

        txt_files = [n for n in all_names if n.lower().endswith(".txt")]
        data_candidates = [n for n in txt_files if not Path(n).name.startswith("_")]
        if not data_candidates:
            data_candidates = txt_files

        # Pick the largest one (most likely the data file)
        data_candidates.sort(key=lambda n: zf.getinfo(n).compress_size, reverse=True)
        chosen = data_candidates[0]
        print(f"  Extracting: {chosen}")

        zf.extract(chosen, raw_dir)
        extracted = raw_dir / chosen
        if not extracted.exists():
            extracted = raw_dir / Path(chosen).name

    print(f"  Data file : {extracted}")
    return str(extracted)


def load_pairs(path, max_len, num_pairs=None):
    pairs = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.lstrip("\ufeff").strip()
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            src_tok = tokenise(parts[0])
            tgt_tok = tokenise(parts[1])
            if 1 <= len(src_tok) <= max_len and 1 <= len(tgt_tok) <= max_len:
                pairs.append((src_tok, tgt_tok))

    print(f"  Read {len(pairs):,} pairs passing length filter (max_len={max_len})")

    if num_pairs:
        random.seed(C.SEED)
        random.shuffle(pairs)
        pairs = pairs[:num_pairs]
    return pairs


# ── PyTorch Dataset ────────────────────────────────────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs     = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_tok, tgt_tok = self.pairs[idx]
        src = torch.tensor([SOS_IDX] + self.src_vocab.encode(src_tok) + [EOS_IDX])
        tgt = torch.tensor([SOS_IDX] + self.tgt_vocab.encode(tgt_tok) + [EOS_IDX])
        return src, tgt


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=False)
    tgt_padded = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=False)
    return src_padded, tgt_padded


# ── Public helper ──────────────────────────────────────────────────────────
def get_data_loaders():
    path  = download_data()
    pairs = load_pairs(path, C.MAX_LEN, C.NUM_PAIRS)

    if len(pairs) == 0:
        raise RuntimeError(
            "No sentence pairs were loaded!\n"
            "Check that the data file downloaded correctly and "
            "MAX_LEN in config.py is not too small."
        )

    print(f"Total sentence pairs after filtering: {len(pairs):,}")

    random.seed(C.SEED)
    random.shuffle(pairs)
    n       = len(pairs)
    n_train = int(0.90 * n)
    n_val   = int(0.05 * n)
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train : n_train + n_val]
    test_pairs  = pairs[n_train + n_val :]

    src_vocab = Vocab().build([p[0] for p in train_pairs], C.MIN_FREQ)
    tgt_vocab = Vocab().build([p[1] for p in train_pairs], C.MIN_FREQ)
    print(f"Vocab sizes  ->  src: {len(src_vocab):,}  |  tgt: {len(tgt_vocab):,}")

    def make_loader(p, shuffle):
        ds = TranslationDataset(p, src_vocab, tgt_vocab)
        return DataLoader(
            ds,
            batch_size=C.BATCH_SIZE,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0,    # 0 = safe on Windows
            pin_memory=False,
        )

    return (
        make_loader(train_pairs, True),
        make_loader(val_pairs,   False),
        make_loader(test_pairs,  False),
        src_vocab,
        tgt_vocab,
    )
