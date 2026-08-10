from __future__ import annotations

from pathlib import Path

from .constants import MODEL_ARCHITECTURE
from .isolated import isolated_script


def runtime_helper_path() -> Path:
    return isolated_script("flux_runtime.py")


def evaluation_script_path(name: str) -> Path:
    allowed = {
        "base": "evaluate_base.py",
        "checkpoint_sweep": "evaluate_checkpoint_sweep.py",
        "scale_sweep": "evaluate_scale_sweep.py",
    }
    return isolated_script(allowed[name])


def expected_architecture() -> str:
    return MODEL_ARCHITECTURE
