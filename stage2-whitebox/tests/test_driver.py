import importlib.util
from pathlib import Path

from s2.runio import require_ac_power


def test_battery_text_still_refuses():
    try:
        require_ac_power("Now drawing from 'Battery Power'\n")
    except RuntimeError as exc:
        assert "not on AC power" in str(exc)
    else:
        raise AssertionError("battery text was accepted")


def test_step_argv_points_at_the_fit_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_checkpoint.py"
    spec = importlib.util.spec_from_file_location("run_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = module.step_argv("fit", "olmo3_32b", "dpo", "cuda", "main")
    assert argv[1].endswith("31_fit.py")
    assert "--device" in argv and "cuda" in argv
    assert "--split" not in module.step_argv("readout", "olmo3_32b", "dpo", "cuda", "main")
    assert "--split" in module.step_argv("nocot", "olmo3_32b", "dpo", "cuda", "main")
