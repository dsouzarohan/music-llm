import json
from pathlib import Path

import pytest
import torch

from src.model.model import MusicTransformer
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.train import accumulation_groups, parse_args, train
from tests.conftest import TINY
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
    assert {"config.json", "metrics.jsonl", "best.pt", "final.pt"} <= {p.name for p in run_dir.iterdir()}
    config = json.loads((run_dir / "config.json").read_text())
    assert config["n_params"] > 0 and config["name"] == "smoke"
    rows = metrics(run_dir)
    assert [r["epoch"] for r in rows] == [0, 1] and [r["step"] for r in rows] == [3, 6]
    assert {"train_loss", "val_loss", "lr", "grad_norm", "tokens_per_sec", "mfu"} <= rows[0].keys()


def test_same_seed_gives_same_val_loss(corpus, tmp_path):
    a = metrics(run(corpus, tmp_path / "a"))
    b = metrics(run(corpus, tmp_path / "b"))
    assert a[-1]["val_loss"] == b[-1]["val_loss"]


def test_resume_continues_exactly(corpus, tmp_path):
    full = run(corpus, tmp_path / "full", "--epochs", "2", "--save_every", "1")
    resumed = run(corpus, tmp_path / "resumed", "--epochs", "2", "--resume", str(full / "epoch_1.pt"))

    a, b = load_checkpoint(full / "final.pt"), load_checkpoint(resumed / "final.pt")
    assert (a["epoch"], a["step"]) == (b["epoch"], b["step"]) == (1, 6)
    assert a["scheduler"]["last_epoch"] == b["scheduler"]["last_epoch"] == 6
    assert a["optimizer"]["param_groups"][0]["lr"] == b["optimizer"]["param_groups"][0]["lr"]
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    assert [r["epoch"] for r in metrics(resumed)] == [1]


def test_compiled_model_checkpoint_loads_into_plain_model(tmp_path):
    torch.manual_seed(0)
    model = MusicTransformer(**TINY)
    compiled = torch.compile(model)
    assert next(iter(compiled.state_dict())).startswith("_orig_mod.")
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=10)
    config = {"vocab_size": TINY["vocab_size"], "embedding_dim": TINY["n_embd"], "n_head": TINY["n_head"],
              "n_layer": TINY["n_layer"], "block_size": TINY["block_size"], "dropout": TINY["dropout"]}
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
