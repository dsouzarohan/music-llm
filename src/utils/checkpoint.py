"""Checkpoint save/load with one schema for best.pt, epoch_N.pt and final.pt."""

import torch

COMPILE_PREFIX = "_orig_mod."


def unwrap(model):
    """Returns the original module behind a torch.compile wrapper (or the model itself)."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(path, *, model, optimizer, scheduler, scaler, epoch, step, val_loss, best_val_loss, config, rng=None):
    """Saves model/optimizer/scheduler/scaler state plus training progress, the run config and RNG state.

    The model state is taken from the uncompiled module, so keys carry no `_orig_mod.` prefix.
    """
    torch.save(
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
