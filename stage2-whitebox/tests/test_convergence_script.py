import subprocess
import sys
from pathlib import Path

import torch


def test_convergence_script_exits_on_a_mismatch(tmp_path):
    small = tmp_path / "small.pt"
    large = tmp_path / "large.pt"
    torch.save({"rlens_readout": torch.tensor([[1.0, 0.0]])}, small)
    torch.save({"rlens_readout": torch.tensor([[0.0, 1.0]])}, large)
    script = Path(__file__).resolve().parents[1] / "scripts" / "33_convergence.py"
    result = subprocess.run(
        [sys.executable, str(script), "--small", str(small), "--large", str(large)],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert "min cosine" in result.stdout
