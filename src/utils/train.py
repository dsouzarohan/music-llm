# train.py

import argparse
import json
import math
import random
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.model import MusicTransformer
from src.utils.checkpoint import (CHECKPOINT_DIR, load_checkpoint, prune_snapshots, resolve_checkpoint, run_dir_of,
                                  save_checkpoint, snapshot_name)
from src.utils.config import apply_config_file
from src.utils.data.guitar_dataset import GuitarDataset
from src.utils.data.random_guitar_seq_dataset import RandomGuitarSeqDataset
from src.utils.metrics import device_peak_flops, flops_per_token, mfu

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


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    """Restores RNG states; tensors may have been moved to a device by torch.load's map_location."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_parser():
    parser = argparse.ArgumentParser(description="Train MusicTransformer. Precedence: CLI flag > --config TOML > defaults.")

    parser.add_argument("--config", type=str, default=None, help="TOML run spec whose keys are these options")
    parser.add_argument("--name", type=str, default="run", help="Run name, used in the run directory name")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--runs_dir", type=str, default=None, help="Parent of run directories (default: {data_root}/runs)")
    parser.add_argument("--run_dir", type=str, default=None, help="Exact run directory (default: {runs_dir}/<YYYY-MM-DD>/<HHMM>-<name>, or the resumed run's directory)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--micro_batch_size", type=int, default=MICRO_BATCH_SIZE)
    parser.add_argument("--train_windows", type=int, default=2000, help="Random training windows per epoch")
    parser.add_argument("--val_windows", type=int, default=512, help="Fixed validation windows, evenly spaced over the val set")
    parser.add_argument("--use_guitar_ds", action="store_true", help="Use GuitarDataset instead of RandomGuitarSeqDataset for training")
    parser.add_argument("--save_every", type=int, default=0, help="Every N epochs, overwrite last.pt and write an epoch_NNN.pt snapshot (0 = last.pt only at the end)")
    parser.add_argument("--keep_last", type=int, default=2, help="Keep this many newest epoch_NNN.pt snapshots (0 = write none); best.pt and last.pt are always kept")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="torch.compile the model")
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--profile_steps", type=int, default=0, help="Profile this many micro-steps of the first epoch into {run_dir}/profile.txt")
    parser.add_argument("--peak_flops", type=float, default=None, help="Device peak FLOP/s for MFU (default: looked up for known GPUs)")

    # Hyperparameters arguments (defaults loaded from src.utils.hyperparameters)
    parser.add_argument("--block_size", type=int, default=BLOCK_SIZE, help="Context window size")
    parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM, help="Embedding dimension (n_embd)")
    parser.add_argument("--n_layer", type=int, default=N_LAYER, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=N_HEAD, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout rate")
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE, help="Vocabulary size")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Peak learning rate")

    # Training run arguments
    parser.add_argument("--resume", type=str, default=None, help="Run directory (resumes its checkpoints/last.pt) or checkpoint file; restores model, optimizer, scheduler, RNG")
    parser.add_argument("--finetune_from", type=str, default=None, help="Checkpoint to load model weights from, with a fresh optimizer")

    return parser


# parses args, can save multiple run configs for this script
def parse_args(argv=None):
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        apply_config_file(parser, pre.config)
    return parser.parse_args(argv)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps') # metal for Apple Silicon
    return torch.device('cpu')


def accumulation_groups(num_micro_batches, grad_accum_steps):
    """Size of the accumulation group each micro-batch belongs to; the last group may be smaller."""
    sizes = []
    for start in range(0, num_micro_batches, grad_accum_steps):
        size = min(grad_accum_steps, num_micro_batches - start)
        sizes += [size] * size
    return sizes


def resolve_run_dir(args, now=None):
    """--run_dir if given, else the resumed run's directory, else a new `{runs_dir}/<YYYY-MM-DD>/<HHMM>-<name>`
    (suffixed `-2`, `-3`, ... if that already exists)."""
    if args.run_dir:
        return Path(args.run_dir)
    if args.resume:
        return run_dir_of(resolve_checkpoint(args.resume, "last.pt"))
    now = now or datetime.now()
    day = (Path(args.runs_dir) if args.runs_dir else Path(args.data_root) / "runs") / f"{now:%Y-%m-%d}"
    base = f"{now:%H%M}-{args.name}"
    run_dir, n = day / base, 1
    while run_dir.exists():
        n += 1
        run_dir = day / f"{base}-{n}"
    return run_dir


def profile_report(prof, device):
    """Where the profiled steps spent time: wall vs CPU vs GPU kernels, the forward "attention"/"mlp"
    regions' share of device time, then the op table."""
    events = prof.key_averages()
    cpu_side = {e.key: e for e in events if e.device_type == torch.autograd.DeviceType.CPU}
    wall = sum(e.cpu_time_total for k, e in cpu_side.items() if k.startswith("ProfilerStep"))
    cpu = sum(e.self_cpu_time_total for e in events)
    kernels = sum(e.self_device_time_total for e in events)
    lines = [
        f"profiled steps: wall {wall / 1e3:.1f} ms, CPU {cpu / 1e3:.1f} ms, device kernels {kernels / 1e3:.1f} ms"
        + (f" (device busy {kernels / wall:.0%} of wall)" if kernels and wall else ""),
        "forward regions (labels vanish under torch.compile, profile with --no-compile):",
    ]
    for region in ("attention", "mlp"):
        e = cpu_side.get(region)
        if e is None:
            continue
        if kernels:
            lines.append(f"  {region:<10} device {e.device_time_total / 1e3:9.1f} ms  {e.device_time_total / kernels:6.1%} of device time")
        else:
            lines.append(f"  {region:<10} CPU {e.cpu_time_total / 1e3:9.1f} ms  {e.cpu_time_total / wall:6.1%} of wall")
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    return "\n".join(lines) + "\n\n" + events.table(sort_by=sort_key, row_limit=30)


def evaluate(model, val_dl, device, autocast):
    model.eval()
    val_loss, total_tokens = 0.0, 0
    with torch.no_grad(), autocast():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            _, l = model(x, y) # loss already mean per-token for batch
            val_loss += l.item() * y.numel()
            total_tokens += y.numel()
    return val_loss / total_tokens


def train(args):
    """Runs training and returns the run directory."""
    set_seed(args.seed)

    data_root = Path(args.data_root)
    train_tok_folder = data_root / "tokenized" / "train-aug"
    val_tok_folder = data_root / "tokenized" / "val"
    run_dir = resolve_run_dir(args)
    ckpt_dir = run_dir / CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    grad_accum_steps = max(1, args.batch_size // args.micro_batch_size) # handles cases if our batch size is not perfectly divisible

    # -------------
    # device and AMP (Automatic Mixed Precision, CUDA only)
    # -------------

    device = pick_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16 # bf16 has fp32's range, so it needs no loss scaling
    autocast = lambda: torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    print(f"Device {device}, AMP {amp_dtype if use_amp else 'off'}, run dir {run_dir}")

    # ---------------------------
    # datasets and dataloaders
    # ---------------------------

    train_files = sorted(train_tok_folder.glob("*.json"))
    val_files = sorted(val_tok_folder.glob("*.json"))
    random.shuffle(train_files)

    if args.use_guitar_ds:
        train_ds = GuitarDataset(block_size=args.block_size, stride=args.block_size // 2, file_list=train_files)
    else:
        train_ds = RandomGuitarSeqDataset(block_size=args.block_size, epoch_len=args.train_windows, file_list=train_files)
    val_ds = GuitarDataset(block_size=args.block_size, stride=args.block_size, file_list=val_files, max_windows=args.val_windows)

    train_dl = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.micro_batch_size * 2, shuffle=False) # no gradients stored, so larger batches fit

    print(f"Train: {len(train_files)} files, {train_ds.total_tokens} tokens, {len(train_ds)} windows/epoch")
    print(f"Val:   {len(val_files)} files, {val_ds.total_tokens} tokens, {len(val_ds)} fixed windows")

    # -------
    # model
    # -------

    model = MusicTransformer(args.vocab_size, args.embedding_dim, args.n_head, args.n_layer, args.block_size, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    flops_tok = flops_per_token(n_params, args.n_layer, args.embedding_dim, args.block_size)
    peak = args.peak_flops or device_peak_flops(device)
    print(f"Parameters {n_params:,}, {flops_tok / 1e9:.2f} GFLOP/token, peak {peak or 'unknown'} FLOP/s")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))

    num_micro_batches = len(train_dl) # number of batches per epoch
    steps_per_epoch = math.ceil(num_micro_batches / grad_accum_steps)
    group_sizes = accumulation_groups(num_micro_batches, grad_accum_steps)

    def make_scheduler(total_steps):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            total_steps=total_steps,
            pct_start=0.1, # Warmup for first 10% of training,
            anneal_strategy='cos', # cosine decay
            div_factor=25.0, # initial LR will be max_lr / 25
            final_div_factor=1000.0, # Final LR will be tiny
        )

    # -------
    # Resume / fine-tune
    # -------
    start_epoch, step, best_val_loss = 0, 0, float("inf")

    if args.resume:
        resume_path = resolve_checkpoint(args.resume, "last.pt")
        ckpt = load_checkpoint(resume_path, map_location="cpu") # load_state_dict copies onto the model/optimizer device
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if ckpt.get("scheduler") is not None:
            scheduler = make_scheduler(steps_per_epoch * args.epochs)
            optimizer.load_state_dict(ckpt["optimizer"]) # after the scheduler, which resets the LR when created
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            print("Checkpoint has no scheduler state (pre-T3); restarting the LR schedule over the remaining epochs")
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler = make_scheduler(steps_per_epoch * (args.epochs - start_epoch))
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("rng") is not None:
            set_rng_state(ckpt["rng"])
        del ckpt
        print(f"Resumed {resume_path} at epoch {start_epoch}, step {step}, best val {best_val_loss:.4f}")
    else:
        if args.finetune_from:
            model.load_state_dict(load_checkpoint(args.finetune_from, map_location="cpu")["model"])
            print(f"Fine-tuning from {args.finetune_from}")
        scheduler = make_scheduler(steps_per_epoch * args.epochs)

    if args.compile:
        print(f"Compiling model for {device.type}")
        model = torch.compile(model)

    # -------
    # run files
    # -------
    config = {**vars(args), "run_dir": str(run_dir), "device": str(device), "n_params": n_params,
              "flops_per_token": flops_tok, "peak_flops": peak, "git_commit": git_commit()}
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    metrics_path = run_dir / "metrics.jsonl"

    def checkpoint(name, epoch, val_loss):
        save_checkpoint(ckpt_dir / name, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        epoch=epoch, step=step, val_loss=val_loss, best_val_loss=best_val_loss, config=config,
                        rng=rng_state())

    print(f"Training epochs {start_epoch}..{args.epochs - 1}: {steps_per_epoch} steps/epoch, {grad_accum_steps} micro-batches/step")

    # --------------------------------
    # training loop and checkpointing
    # --------------------------------
    val_loss = None
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, grad_norms, tokens = 0.0, [], 0

        profiling = args.profile_steps > 0 and epoch == start_epoch
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=args.profile_steps, repeat=1),
        ) if profiling else nullcontext()

        t0 = time.perf_counter()
        with prof:
            for i, (x, y) in enumerate(train_dl):
                x, y = x.to(device), y.to(device)

                with autocast():
                    _, loss = model(x, y)
                # scale by the size of this accumulation group so gradients average correctly
                scaler.scale(loss / group_sizes[i]).backward()
                loss_sum += loss.item()
                tokens += x.numel()

                # Step optimizer at the end of each accumulation group
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == num_micro_batches:
                    scaler.unscale_(optimizer)
                    grad_norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    # Resets gradients ONLY after the step, or else we will lose our gradients
                    optimizer.zero_grad(set_to_none=True)
                    step += 1

                if profiling:
                    prof.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.perf_counter() - t0

        if profiling:
            (run_dir / "profile.txt").write_text(profile_report(prof, device))
            print(f"Wrote {run_dir / 'profile.txt'}")

        # ----- validate ------
        val_loss = evaluate(model, val_dl, device, autocast)
        tokens_per_sec = tokens / train_time
        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": loss_sum / num_micro_batches,
            "val_loss": val_loss,
            "val_ppl": math.exp(val_loss),
            "val_bits_per_token": val_loss / math.log(2),
            "lr": scheduler.get_last_lr()[0],
            "grad_norm": sum(grad_norms) / len(grad_norms),
            "tokens_per_sec": tokens_per_sec,
            "mfu": mfu(tokens_per_sec, flops_tok, peak),
            "train_time_s": train_time,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch} step {step} train {row['train_loss']:.4f} val_loss {val_loss:.4f} ppl {row['val_ppl']:.1f} "
            f"lr {row['lr']:.2e} grad_norm {row['grad_norm']:.3f} tok/s {tokens_per_sec:,.0f}"
            + (f" mfu {row['mfu']:.1%}" if row["mfu"] is not None else "")
        )

        # ---------------
        # checkpointing
        # ---------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint("best.pt", epoch, val_loss)
            print(f"Saved new best checkpoint (val_loss={best_val_loss:.4f})")
        periodic = args.save_every > 0 and (epoch + 1) % args.save_every == 0
        if periodic or epoch == args.epochs - 1:
            checkpoint("last.pt", epoch, val_loss)
        if periodic and args.keep_last > 0:
            checkpoint(snapshot_name(epoch + 1), epoch, val_loss)
            for p in prune_snapshots(ckpt_dir, args.keep_last):
                print(f"Deleted old snapshot {p.name}")
    print("Training finished")
    return run_dir


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
