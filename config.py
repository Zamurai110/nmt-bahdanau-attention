"""
Central configuration for the NMT project.
Edit these values before training.
"""

import torch

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_URL   = "https://www.manythings.org/anki/fra-eng.zip"  # French-English pairs
DATA_DIR   = "data/raw"
PROCESSED  = "data/processed"
CHECKPOINTS= "checkpoints"
LOGS       = "logs"

# ── Dataset ────────────────────────────────────────────────────────────────
SRC_LANG   = "en"
TGT_LANG   = "fr"
MAX_LEN    = 20          # max tokens per sentence (filters longer pairs)
MIN_FREQ   = 2           # minimum word frequency to keep in vocab
NUM_PAIRS  = None        # None = use all; set e.g. 50_000 to cap dataset size

# ── Model ──────────────────────────────────────────────────────────────────
EMB_DIM    = 256
HID_DIM    = 512
ENC_LAYERS = 2
DEC_LAYERS = 2
ENC_DROPOUT= 0.3
DEC_DROPOUT= 0.3

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE     = 128
NUM_EPOCHS     = 30
LR             = 3e-4
CLIP           = 1.0          # gradient clipping max-norm
TEACHER_FORCING= 0.5          # probability of using teacher forcing each step
PATIENCE       = 5            # early-stop patience (epochs without improvement)

# ── Beam Search ────────────────────────────────────────────────────────────
BEAM_SIZE      = 5
BEAM_ALPHA     = 0.75         # length-penalty exponent

# ── Misc ───────────────────────────────────────────────────────────────────
SEED     = 42
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
