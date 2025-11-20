# train.py

import os
import math
import random
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.model import MusicTransformer
from src.utils.data.guitar_dataset import GuitarDataset
from src.utils.data.random_guitar_seq_dataset import RandomGuitarSeqDataset

from src.utils.hyperparameters import (
    BLOCK_SIZE,
    BATCH_SIZE,
    EMBEDDING_DIM,
    N_LAYER,
    N_HEAD,
    DROPOUT,
    VOCAB_SIZE,
    LEARNING_RATE
)

# -------
# utils
# -------

# Sets seeds for all libraries being used (to reproduce if required)
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

# parses args, can save multiple run configs for this script
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../../data/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--checkpoint_dir", type=str, default="../../data/checkpoints")
    parser.add_argument("--use_guitar_ds", action="store_true", help="Use GuitarDataset instead of RandomGuitarWindowDataset")
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0 = only best + last)")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # -------------
    # define paths
    # -------------

    data_root = Path(args.data_root)
    midi_folder = data_root / "midi/"
    augmented_folder = data_root / "augmented/"
    tokenized_folder = data_root / "tokenized/"
    splits_folder = data_root / "splits/"
    train_tok_folder = tokenized_folder / "train-aug/"
    val_tok_folder = tokenized_folder / "val/"
    train_midi_folder = data_root / "train_midi/"
    val_midi_folder = data_root / "val-midi/"

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ----------------------
    # define hyperparameters
    # ----------------------

    batch_size = args.batch_size

    block_size = BLOCK_SIZE
    n_embd = EMBEDDING_DIM
    vocab_size = VOCAB_SIZE
    n_layer = N_LAYER
    n_head = N_HEAD
    dropout = DROPOUT

    learning_rate = LEARNING_RATE

    # -------------
    # device
    # -------------

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps') # metal for Apple Silicon
    else:
        device = torch.device('cpu')

    # ---------------------------------
    # AMP (Automatic Mixed Precision)
    # ---------------------------------

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    use_amp = (device.type == "cuda")
    if use_amp and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        # uses brainfloat 16
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    print("Using device:", device)
    print("AMP enabled?", use_amp, "dtype: ", amp_dtype)

    # ---------------------------
    # datasets and dataloaders
    # ---------------------------

    train_files = sorted(train_tok_folder.glob("*.json"))
    val_files = sorted(val_tok_folder.glob("*.json"))

    print(f"{len(train_files)} training files")
    print(f"{len(val_files)} validation files")

    random.shuffle(train_files)

    if args.use_guitar_ds:
        print("Using GuitarDataset (not recommended)")
        train_ds = GuitarDataset(block_size=block_size, stride=block_size // 2, file_list=train_files)
        val_ds = GuitarDataset(block_size=block_size, stride=block_size // 2, file_list=val_files)
    else:
        print("Using RandomGuitarWindowDataset")
        print("block size", block_size)
        train_ds = RandomGuitarSeqDataset(block_size=block_size, epoch_len=2000, file_list=train_files)
        val_ds = RandomGuitarSeqDataset(block_size=block_size, epoch_len=400, file_list=val_files)


    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=True, drop_last=True)

    print("Training tokens count:", train_dl.dataset.total_tokens)
    print("Validation tokens count:", val_dl.dataset.total_tokens)

    # -------
    # model
    # -------

    model = MusicTransformer(
        vocab_size, n_embd, n_head, n_layer, block_size, dropout
    ).to(device)

    # AdamW and scaler
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    print("Parameter count:", sum(p.numel() for p in model.parameters()))
    print("Training on device:", device)
    print("Using amp?", use_amp, amp_dtype)

    # --------------------------------
    # training loop and checkpointing
    # --------------------------------

    epochs = args.epochs
    V = vocab_size
    lnV = np.log(V)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # train
        for x, y in train_dl:
            x , y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none = True)

            # using torch.autocast here with device_Type to avoid backend specific contexts
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits, loss = model(x, y)

            if use_amp and amp_dtype == torch.float16:
                # FP16 path: scale, unscale before clopping, then step
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            else:
                # BF16 / no-amp path:
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
                optimizer.step()

        # ----- validate ------

        model.eval()
        val_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            # also using AMP in eval to cut memory/latency
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    _, l = model(x, y) # loss already mean per-token for batch
                    num_tokens = y.numel()
                    val_loss += l.item() * num_tokens
                    total_tokens += num_tokens

        avg_loss = val_loss / total_tokens
        ppl = np.exp(avg_loss)
        bpc = avg_loss / np.log(2)
        improv_ratio = V / ppl
        delta_nats = lnV - avg_loss

        # TODO: understand these metrics in depth
        print(
            f"epoch {epoch:03d} "
            f"train {loss.item():.4f} "
            f"val_loss {avg_loss:.4f}  ppl {ppl:.0f}  "
            f"bpc {bpc:.3f}  Δnats {delta_nats:.3f}  x-better {improv_ratio:.2f}x  (lnV {lnV:.3f})"
        )

        # ---------------
        # checkpointing
        # ---------------

        # best model
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if (use_amp and amp_dtype == torch.float16) else None,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "vocab_size": vocab_size,
                "hyperparams": {
                    "n_embd": n_embd,
                    "n_head": n_head,
                    "n_layer": n_layer,
                    "block_size": block_size,
                    "dropout": dropout
                },
            }
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"Saved new best checkpoint (val_loss={best_val_loss:.4f})")

        # periodic checkpoints
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if (use_amp and amp_dtype == torch.float16) else None,
                    "epoch": epoch,
                    "val_loss": avg_loss,
                },
                os.path.join(args.checkpoint_dir, f"epoch_{epoch+1:.03d}.pt"),
            )
            print(f"Saved checkpoint at epoch {epoch+1}")

    # saving final checkpoint
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if (use_amp and amp_dtype == torch.float16) else None,
            "epoch": epochs - 1
        },
        os.path.join(args.checkpoint_dir, "final.pt"),
    )
    print("Training finished")

if __name__ == "__main__":
    main()
