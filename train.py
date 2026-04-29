"""
Training loop with:
  - Teacher forcing
  - Gradient clipping
  - Checkpoint saving (best val loss)
  - Early stopping
  - TensorBoard logging
"""

import os, time, math, pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import config as C
from data.dataset   import get_data_loaders, PAD_IDX
from model.encoder  import Encoder
from model.decoder  import Decoder
from model.seq2seq  import Seq2Seq

torch.manual_seed(C.SEED)


# ── Build model ────────────────────────────────────────────────────────────
def build_model(src_vocab_size: int, tgt_vocab_size: int) -> Seq2Seq:
    enc = Encoder(src_vocab_size, C.EMB_DIM, C.HID_DIM,
                  C.ENC_LAYERS, C.ENC_DROPOUT)
    dec = Decoder(tgt_vocab_size, C.EMB_DIM,
                  C.HID_DIM * 2,    # bi-GRU output
                  C.HID_DIM, C.DEC_DROPOUT)
    model = Seq2Seq(enc, dec, C.DEVICE).to(C.DEVICE)

    # Xavier uniform init
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


# ── One epoch ──────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, clip, train: bool):
    model.train() if train else model.eval()
    total_loss = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for src, trg in loader:
            src, trg = src.to(C.DEVICE), trg.to(C.DEVICE)

            if train:
                optimizer.zero_grad()

            output = model(src, trg)
            # output : (trg_len-1, B, vocab)
            # trg    : (trg_len, B)
            out_flat = output.reshape(-1, output.shape[-1])
            trg_flat = trg[1:].reshape(-1)

            loss = criterion(out_flat, trg_flat)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    Path(C.CHECKPOINTS).mkdir(parents=True, exist_ok=True)
    Path(C.LOGS).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_data_loaders()

    # Save vocabularies for inference
    with open(f"{C.CHECKPOINTS}/src_vocab.pkl", "wb") as f:
        pickle.dump(src_vocab, f)
    with open(f"{C.CHECKPOINTS}/tgt_vocab.pkl", "wb") as f:
        pickle.dump(tgt_vocab, f)

    model    = build_model(len(src_vocab), len(tgt_vocab))
    print(f"Model parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = Adam(model.parameters(), lr=C.LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    writer    = SummaryWriter(C.LOGS)

    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, C.NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, C.CLIP, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, C.CLIP, train=False)
        elapsed    = time.time() - t0

        scheduler.step(val_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("PPL/train",  math.exp(train_loss), epoch)
        writer.add_scalar("PPL/val",    math.exp(val_loss),   epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f}  PPL: {math.exp(train_loss):7.2f} | "
            f"Val Loss: {val_loss:.4f}  PPL: {math.exp(val_loss):7.2f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    val_loss,
                "src_vocab_size": len(src_vocab),
                "tgt_vocab_size": len(tgt_vocab),
            }, f"{C.CHECKPOINTS}/best_model.pt")
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= C.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    writer.close()
    print("\nTraining complete. Best val loss:", round(best_val_loss, 4))


if __name__ == "__main__":
    main()
