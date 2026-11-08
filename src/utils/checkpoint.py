"""Checkpoint save/load with one schema for best.pt, last.pt and epoch_NNN.pt.

A run directory keeps its checkpoints in `<run_dir>/checkpoints/`.
"""

import os
import re
from pathlib import Path

import torch

COMPILE_PREFIX = "_orig_mod."
CHECKPOINT_DIR = "checkpoints"
SNAPSHOT_RE = re.compile(r"epoch_(\d+)\.pt")


def unwrap(model):
    """Returns the original module behind a torch.compile wrapper (or the model itself)."""
    return getattr(model, "_orig_mod", model)


def snapshot_name(epochs_done):
    """`epoch_NNN.pt` for a snapshot taken after `epochs_done` epochs."""
    return f"epoch_{epochs_done:03d}.pt"


def atomic_save(obj, path):
    """torch.save to `<path>.tmp`, fsync, then os.replace onto `path`.

    `path` is either the previous complete file or the new complete file, never a partial one.
    On an exception the temp file is removed and the exception propagates.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_checkpoint(path, *, model, optimizer, scheduler, scaler, epoch, step, val_loss, best_val_loss, config, rng=None):
    """Atomically saves model/optimizer/scheduler/scaler state plus training progress, the run config and RNG state.

    The model state is taken from the uncompiled module, so keys carry no `_orig_mod.` prefix.
    """
    atomic_save(
        {
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "vocab_size": config["vocab_size"],
            "hyperparams": model_hyperparams(config),
            "config": config,
            "rng": rng,
        },
        path,
    )


def prune_snapshots(ckpt_dir, keep):
    """Deletes all but the `keep` newest `epoch_NNN.pt` files in `ckpt_dir`; returns the deleted paths.

    Other files (best.pt, last.pt) are never touched.
    """
    snapshots = sorted(
        (int(m.group(1)), p) for p in Path(ckpt_dir).iterdir() if (m := SNAPSHOT_RE.fullmatch(p.name))
    )
    doomed = [p for _, p in snapshots[:max(0, len(snapshots) - keep)]]
    for p in doomed:
        p.unlink()
    return doomed


def resolve_checkpoint(path, name):
    """A checkpoint file as is, or `<path>/checkpoints/<name>` when `path` is a run directory."""
    path = Path(path)
    if path.is_dir():
        path = path / CHECKPOINT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}")
    return path


def run_dir_of(ckpt_path):
    """The run directory a checkpoint file belongs to: the parent of `checkpoints/`, or the
    checkpoint's own folder for runs saved before checkpoints moved into a subfolder."""
    parent = Path(ckpt_path).parent
    return parent.parent if parent.name == CHECKPOINT_DIR else parent


def load_checkpoint(path, map_location="cpu"):
    """Loads a checkpoint and strips the `_orig_mod.` prefix older compiled runs saved."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    ckpt["model"] = {k.removeprefix(COMPILE_PREFIX): v for k, v in ckpt["model"].items()}
    return ckpt


def model_hyperparams(config):
    """MusicTransformer constructor kwargs (except vocab_size) from a run config."""
    return {
        "n_embd": config["embedding_dim"],
        "n_head": config["n_head"],
        "n_layer": config["n_layer"],
        "block_size": config["block_size"],
        "dropout": config["dropout"],
    }
