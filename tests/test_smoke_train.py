import json
import math
import subprocess
from pathlib import Path

import pytest
import torch

from src.model.model import MusicTransformer

REPO = Path(__file__).resolve().parent.parent
VOCAB = 64


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """Runs scripts/smoke-train.sh once (compiled, default device); returns (run_dir, stdout)."""
    run_dir = tmp_path_factory.mktemp("run")
    proc = subprocess.run(
        [REPO / "scripts" / "smoke-train.sh", "--run_dir", str(run_dir)],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return run_dir, proc.stdout


def test_smoke_train_completes(smoke_run):
    run_dir, _ = smoke_run
    assert (run_dir / "checkpoints" / "best.pt").exists() and (run_dir / "checkpoints" / "last.pt").exists()
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    val_loss = rows[-1]["val_loss"]
    assert math.isfinite(val_loss) and val_loss < math.log(VOCAB) + 0.5


def test_checkpoint_loads_into_uncompiled_model(smoke_run):
    run_dir, _ = smoke_run
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model = MusicTransformer(vocab_size=state["vocab_size"], **state["hyperparams"])
    model.load_state_dict(state["model"])
