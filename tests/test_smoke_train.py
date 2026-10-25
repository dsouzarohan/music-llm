import math
import re
import subprocess
from pathlib import Path

import pytest
import torch

from src.model.model import MusicTransformer

REPO = Path(__file__).resolve().parent.parent
VOCAB = 64


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """Runs scripts/smoke-train.sh once; returns (checkpoint_dir, stdout)."""
    ckpt = tmp_path_factory.mktemp("ckpt")
    proc = subprocess.run(
        [REPO / "scripts" / "smoke-train.sh", "--checkpoint_dir", str(ckpt)],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return ckpt, proc.stdout


def test_smoke_train_completes(smoke_run):
    ckpt, out = smoke_run
    assert (ckpt / "best.pt").exists() and (ckpt / "final.pt").exists()
    val_loss = float(re.search(r"val_loss (\S+)", out).group(1))
    assert math.isfinite(val_loss) and val_loss < math.log(VOCAB) + 0.5


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason="#3 compiled checkpoints carry an _orig_mod. prefix")
def test_checkpoint_loads_into_uncompiled_model(smoke_run):
    ckpt, _ = smoke_run
    state = torch.load(ckpt / "best.pt", map_location="cpu")
    model = MusicTransformer(vocab_size=state["vocab_size"], **state["hyperparams"])
    model.load_state_dict(state["model"])
