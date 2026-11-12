import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest
import torch

from src.model.model import MusicTransformer
from src.utils import checkpoint as ckpt_mod
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.train import accumulation_groups, parse_args, resolve_run_dir, train
from tests.conftest import MODERN, OLD_BLOCK, TINY, tiny_config
from tests.synthetic import write_corpus

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    write_corpus(root)
    return root


def run(corpus, run_dir, *extra):
    """Tiny CPU training run: 10 micro-batches of 4 per epoch, 4 per step, so 3 steps (4, 4, 2)."""
    args = parse_args([
        "--config", str(REPO / "configs" / "smoke.toml"), "--data_root", str(corpus), "--run_dir", str(run_dir),
        "--device", "cpu", "--no-compile", "--train_windows", "40", "--micro_batch_size", "4", "--batch_size", "16",
        "--val_windows", "8", *extra,
    ])
    return train(args)


def metrics(run_dir):
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_accumulation_groups_cover_partial_last_group():
    assert accumulation_groups(10, 4) == [4] * 8 + [2] * 2
    assert accumulation_groups(8, 4) == [4] * 8


def test_config_file_then_cli_precedence(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('n_layer = 3\nname = "from-toml"\n')
    args = parse_args(["--config", str(cfg), "--name", "from-cli"])
    assert (args.n_layer, args.name) == (3, "from-cli")


def test_config_file_rejects_unknown_keys(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text("n_layers = 3\n")
    with pytest.raises(ValueError, match="n_layers"):
        parse_args(["--config", str(cfg)])


def test_run_dir_has_config_metrics_and_checkpoints(corpus, tmp_path):
    run_dir = run(corpus, tmp_path / "r", "--epochs", "2")
    assert {"config.json", "metrics.jsonl", "checkpoints"} <= {p.name for p in run_dir.iterdir()}
    assert {p.name for p in (run_dir / "checkpoints").iterdir()} == {"best.pt", "last.pt"}
    config = json.loads((run_dir / "config.json").read_text())
    assert config["n_params"] > 0 and config["name"] == "smoke"
    rows = metrics(run_dir)
    assert [r["epoch"] for r in rows] == [0, 1] and [r["step"] for r in rows] == [3, 6]
    assert {"train_loss", "val_loss", "lr", "grad_norm", "tokens_per_sec", "mfu"} <= rows[0].keys()


def test_same_seed_gives_same_val_loss(corpus, tmp_path):
    a = metrics(run(corpus, tmp_path / "a"))
    b = metrics(run(corpus, tmp_path / "b"))
    assert a[-1]["val_loss"] == b[-1]["val_loss"]


def test_resume_from_run_dir_continues_exactly(corpus, tmp_path):
    full = run(corpus, tmp_path / "full", "--epochs", "2", "--save_every", "1")
    # a run interrupted after epoch 0: its last.pt is full's epoch-1 snapshot
    resumed = tmp_path / "resumed"
    (resumed / "checkpoints").mkdir(parents=True)
    shutil.copy(full / "checkpoints" / "epoch_001.pt", resumed / "checkpoints" / "last.pt")
    run(corpus, resumed, "--epochs", "2", "--resume", str(resumed))

    a, b = load_checkpoint(full / "checkpoints" / "last.pt"), load_checkpoint(resumed / "checkpoints" / "last.pt")
    assert (a["epoch"], a["step"]) == (b["epoch"], b["step"]) == (1, 6)
    assert a["scheduler"]["last_epoch"] == b["scheduler"]["last_epoch"] == 6
    assert a["optimizer"]["param_groups"][0]["lr"] == b["optimizer"]["param_groups"][0]["lr"]
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    assert [r["epoch"] for r in metrics(resumed)] == [1]


def test_keep_last_leaves_best_last_and_newest_snapshots(corpus, tmp_path):
    run_dir = run(corpus, tmp_path / "r", "--epochs", "3", "--save_every", "1", "--keep_last", "2")
    names = {p.name for p in (run_dir / "checkpoints").iterdir()}
    assert names == {"best.pt", "last.pt", "epoch_002.pt", "epoch_003.pt"}
    assert load_checkpoint(run_dir / "checkpoints" / "last.pt")["epoch"] == 2


def test_run_dir_layout_by_date_and_resume_reuses_it(tmp_path):
    now = datetime(2026, 11, 8, 14, 5)
    args = parse_args(["--runs_dir", str(tmp_path), "--name", "x"])
    first = resolve_run_dir(args, now)
    assert first == tmp_path / "2026-11-08" / "1405-x"
    (first / "checkpoints").mkdir(parents=True)
    (first / "checkpoints" / "last.pt").touch()
    assert resolve_run_dir(args, now) == tmp_path / "2026-11-08" / "1405-x-2"
    for resume in (first, first / "checkpoints" / "last.pt"):
        assert resolve_run_dir(parse_args(["--resume", str(resume)])) == first


def tiny_checkpoint(path, epoch):
    model = MusicTransformer(**TINY)
    optimizer = torch.optim.AdamW(model.parameters())
    config = tiny_config()
    save_checkpoint(path, model=model, optimizer=optimizer,
                    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=10),
                    scaler=torch.amp.GradScaler("cpu", enabled=False), epoch=epoch, step=0, val_loss=1.0,
                    best_val_loss=1.0, config=config)


def test_interrupted_save_keeps_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    tiny_checkpoint(path, epoch=0)
    real_save = torch.save

    def dies_mid_write(obj, f):
        real_save(obj, f)
        f.seek(f.tell() // 2)
        f.truncate()
        raise KeyboardInterrupt

    monkeypatch.setattr(ckpt_mod.torch, "save", dies_mid_write)
    with pytest.raises(KeyboardInterrupt):
        tiny_checkpoint(path, epoch=1)
    monkeypatch.undo()

    assert load_checkpoint(path)["epoch"] == 0
    assert [p.name for p in tmp_path.iterdir()] == ["last.pt"]


def test_compiled_model_checkpoint_loads_into_plain_model(tmp_path):
    torch.manual_seed(0)
    model = MusicTransformer(**TINY, **OLD_BLOCK)
    compiled = torch.compile(model)
    assert next(iter(compiled.state_dict())).startswith("_orig_mod.")
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=10)
    config = tiny_config()
    save_checkpoint(tmp_path / "c.pt", model=compiled, optimizer=optimizer, scheduler=scheduler,
                    scaler=torch.amp.GradScaler("cpu", enabled=False), epoch=0, step=0, val_loss=1.0,
                    best_val_loss=1.0, config=config)

    state = torch.load(tmp_path / "c.pt", weights_only=False)
    plain = MusicTransformer(vocab_size=state["vocab_size"], **state["hyperparams"])
    plain.load_state_dict(state["model"])


def test_load_checkpoint_strips_legacy_prefix(tmp_path):
    model = MusicTransformer(**TINY)
    torch.save({"model": {"_orig_mod." + k: v for k, v in model.state_dict().items()}}, tmp_path / "old.pt")
    MusicTransformer(**TINY).load_state_dict(load_checkpoint(tmp_path / "old.pt")["model"])



def test_rng_state_restore_moves_tensors_to_cpu():
    """map_location=device puts the saved RNG tensor on the GPU; set_rng_state must call .cpu() on it."""
    from unittest.mock import MagicMock
    from src.utils.train import rng_state, set_rng_state
    state = rng_state()
    expected = torch.rand(3)
    on_device = MagicMock()
    on_device.cpu.return_value = state["torch"]
    set_rng_state({**state, "torch": on_device})
    assert torch.equal(torch.rand(3), expected)


def test_modern_block_trains_and_reloads_from_checkpoint(corpus, tmp_path):
    run_dir = run(corpus, tmp_path / "r", "--pos", "rope", "--norm", "rmsnorm", "--mlp", "swiglu",
                  "--qk_norm", "--tie_embeddings")
    state = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    assert state["hyperparams"]["pos"] == "rope" and state["hyperparams"]["tie_embeddings"] is True
    model = MusicTransformer(vocab_size=state["vocab_size"], **state["hyperparams"])
    model.load_state_dict(state["model"])
    assert model.last.weight is model.token_embeddings.weight


def test_default_block_is_modern_in_cli_and_model():
    """T5 picked the full modern block: train.py's defaults and MusicTransformer's agree on it."""
    args = parse_args([])
    cli = {k: getattr(args, k) for k in ("pos", "norm", "mlp", "qk_norm", "tie_embeddings")}
    assert cli == {"pos": "rope", **MODERN}
    model = MusicTransformer(**TINY)
    assert {k: getattr(model, k) for k in cli} == cli


def test_device_kernel_time_counts_each_kernel_once():
    """CPU ops repeat their kernels' time as self device time, and annotations span them; neither is counted."""
    from types import SimpleNamespace as E

    from src.utils.train import device_kernel_time
    cpu, cuda = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
    events = [
        E(device_type=cpu, is_user_annotation=False, self_device_time_total=30),   # aten::mm, launched the kernel
        E(device_type=cuda, is_user_annotation=False, self_device_time_total=30),  # the gemm kernel
        E(device_type=cuda, is_user_annotation=False, self_device_time_total=5),   # a memcpy
        E(device_type=cuda, is_user_annotation=True, self_device_time_total=50),   # ProfilerStep#3 on the GPU
    ]
    assert device_kernel_time(events) == 35
