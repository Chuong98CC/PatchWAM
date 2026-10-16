"""The RobotWin C2R fine-tuning launcher and its initialization selection."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY / "scripts" / "train.sh"
DEFAULT_CHECKPOINT = "Pretrained_models/patchwam-robotwin/step_140000.pt"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="The launcher needs bash")


def launch(tmp_path, **environment):
    """Run the launcher against a stub accelerate that records its arguments."""
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    accelerate = stub / "accelerate"
    accelerate.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@"\n')
    accelerate.chmod(0o755)
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPOSITORY, capture_output=True, text=True, check=True,
        env={**os.environ, "PATH": f"{stub}{os.pathsep}{os.environ['PATH']}", **environment},
    )
    return result.stdout.splitlines()


def test_launcher_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_launcher_selects_initialize_by_default_and_keeps_the_recipe(tmp_path):
    arguments = launch(tmp_path)
    assert arguments[:1] == ["launch"]
    assert "--multi_gpu" in arguments and "--num_processes" in arguments
    assert arguments[arguments.index("--num_processes") + 1] == "4"
    assert arguments[arguments.index("--config") + 1] == "configs/robotwin_vlm_5tasks.yaml"
    assert arguments[arguments.index("--initialize") + 1] == DEFAULT_CHECKPOINT
    assert "--resume" not in arguments
    assert {"model.dit_lora=true", "model.dit_lora_rank=64", "model.dit_lora_alpha=128",
            "model.dit_lora_dropout=0.05"}.issubset(arguments)
    settings = OmegaConf.load(REPOSITORY / "configs/robotwin_vlm_5tasks.yaml")
    assert 4 * settings.training.batch_size * settings.training.accumulation == 256
    assert settings.training.global_batch_size == 256


def test_launcher_switches_to_resume_without_also_initializing(tmp_path):
    arguments = launch(tmp_path, RESUME_CHECKPOINT="/shared/runs/robotwin/step_0005000")
    assert arguments[arguments.index("--resume") + 1] == "/shared/runs/robotwin/step_0005000"
    assert "--initialize" not in arguments
    assert arguments[arguments.index("--num_processes") + 1] == "4"


def test_launcher_accepts_an_explicit_initialization_checkpoint(tmp_path):
    arguments = launch(tmp_path, PRETRAINED_MODEL="/data/other/step_0000001.pt")
    assert arguments[arguments.index("--initialize") + 1] == "/data/other/step_0000001.pt"
    assert "--resume" not in arguments
