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
    LEARNING_RATE,
    MICRO_BATCH_SIZE
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
    parser.add_argument("--micro_batch_size", type=int, default=MICRO_BATCH_SIZE)
    parser.add_argument("--checkpoint_dir", type=str, default="../../data/checkpoints")
    parser.add_argument("--use_guitar_ds", action="store_true", help="Use GuitarDataset instead of RandomGuitarWindowDataset")
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0 = only best + last)")
    parser.add_argument("--seed", type=int, default=42)

    # Hyperparameters arguments (defaults loaded from src.utils.hyperparameters)
    parser.add_argument("--block_size", type=int, default=BLOCK_SIZE, help="Context window size")
    parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM, help="Embedding dimension (n_embd)")
    parser.add_argument("--n_layer", type=int, default=N_LAYER, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=N_HEAD, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout rate")
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE, help="Vocabulary size")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate")

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

    # Now using values from args (which default to the imports if not specified)
    batch_size = args.batch_size
    block_size = args.block_size
    n_embd = args.embedding_dim
    vocab_size = args.vocab_size
    n_layer = args.n_layer
    n_head = args.n_head
    dropout = args.dropout
    learning_rate = args.learning_rate
    micro_batch_size = args.micro_batch_size
    grad_accum_steps = batch_size // micro_batch_size

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


    # Use the batch_size variable from args or default
    train_dl = DataLoader(train_ds, batch_size=micro_batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=micro_batch_size * 2, shuffle=True, drop_last=True) # validation dataset loader can handle slightly larger batches since no gradients are stored

    print("Training tokens count:", train_dl.dataset.total_tokens)
    print("Validation tokens count:", val_dl.dataset.total_tokens)

    # -------
    # model
    # -------

    model = MusicTransformer(
        vocab_size, n_embd, n_head, n_layer, block_size, dropout
    ).to(device)
    print("Launching MusicTransformer model with following hyperparameters")
    print("Vocabulary (number of different tokens)", vocab_size)
    print("Token embedding dimensionality", n_embd)
    print("Number of layers", n_layer)
    print("Number of attention heads", n_head)
    print("Dropout", dropout)
    print(f"Setting Transformer context window to {block_size}, with {batch_size} batch(es) and with a micro batch size of {micro_batch_size}")

    if hasattr(torch, "compile"):
        print("Compiling model for GPU")
        model = torch.compile(model)

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

    num_micro_batches = len(train_dl)
    steps_per_epoch = num_micro_batches // grad_accum_steps
    if num_micro_batches % grad_accum_steps != 0:
        steps_per_epoch += 1
        
    total_steps = steps_per_epoch * epochs

    print(f"Total optimization step: {total_steps}")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.1, # Warmup for first 5% of training,
        anneal_strategy='cos', # cosine decay
        div_factor=25.0, # initial LR will be max_lr / 25
        final_div_factor=1000.0, # Final LR will be tiny
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss_accum = 0.0
        for step, (x, y) in enumerate(train_dl):
            x , y = x.to(device), y.to(device)

            # Using torch.autocast here with device_Type to avoid backend specific contexts
            # Forward pass with autocast
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps # scale loss by accumulation steps so gradients average out correctly

            # Backward pass (accumulate the gradients)
            if use_amp and amp_dtype == torch.float16:
                # FP16 path: just scale
                # optimiser steps will be done at the end
                scaler.scale(loss).backward()
            else:
                # BF16 / no-amp path:
                loss.backward()

            # Accumulate loss
            train_loss_accum += loss.item() * grad_accum_steps

            # Step optimizer (only even N steps)
            is_last_step = (step + 1) == len(train_dl)
            if (step + 1) % grad_accum_steps == 0 or is_last_step:
                if use_amp and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
                    optimizer.step()

                # step the scheduler
                scheduler.step()

                # Resets gradients ONLY after the step, or else we will lose our gradients
                optimizer.zero_grad(set_to_none=True)

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

        num_train_steps = len(train_dl)
        avg_train_loss = train_loss_accum / num_train_steps
        current_lr = scheduler.get_last_lr()[0]

        # TODO: understand these metrics in depth
        print(
            f"epoch {epoch} "
            f"train {avg_train_loss:.4f} "
            f"val_loss {avg_loss:.4f}  ppl {ppl:.0f}  "
            f"bpc {bpc:.3f}  Δnats {delta_nats:.3f}  x-better {improv_ratio:.2f}x  (lnV {lnV:.3f})"
            f"LR {current_lr:.6f}"
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
                    "vocab_size": vocab_size,
                    "hyperparams": {
                        "n_embd": n_embd,
                        "n_head": n_head,
                        "n_layer": n_layer,
                        "block_size": block_size,
                        "dropout": dropout
                    },
                },
                os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pt"),
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
