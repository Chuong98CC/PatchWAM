"""Config units for the five-task RoboTwin-Clean recipe."""

import inspect
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from patchwam.configuration import construct, read_configuration
from patchwam.data import EpisodeDataset

FIVE_TASKS = ["turn_switch", "rotate_qrcode", "blocks_ranking_size",
              "move_stapler_pad", "hanging_mug"]
PARENT = "/data/RoboTwin/RoboTwin-Clean"
FIVE_ROOTS = [f"{PARENT}/{task}" for task in FIVE_TASKS]
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
ENVIRONMENT = {
    "PATCHWAM_TRANSFORMER": "/assets/transformer.safetensors",
    "PATCHWAM_AUTOENCODER": "/assets/autoencoder.safetensors",
    "PATCHWAM_VLM": "/assets/text_encoder",
    "FLUX2_SRC": "/assets/flux2",
    "PATCHWAM_DATA_ROOT": PARENT,
    "PATCHWAM_NORM_STATS": "/data/statistics.json",
    "PATCHWAM_NONIDLE_FILTER": "/data/nonidle.json",
}


@pytest.fixture
def environment(monkeypatch):
    for name, value in ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    return ENVIRONMENT


def test_five_task_config_is_the_source_recipe_plus_the_task_roots(environment):
    source = OmegaConf.to_container(
        read_configuration(CONFIG_DIR / "robotwin_vlm.yaml"), resolve=True)
    subset = OmegaConf.to_container(
        read_configuration(CONFIG_DIR / "robotwin_vlm_5tasks.yaml"), resolve=True)
    assert list(subset["data"]["dataset_dirs"]) == FIVE_ROOTS
    assert "task_selection" not in subset["data"]
    assert subset["training"]["output_dir"] == "runs/robotwin_vlm_5tasks"
    subset["training"]["output_dir"] = source["training"]["output_dir"]
    subset["data"]["dataset_dirs"] = source["data"]["dataset_dirs"]
    subset["data"]["nonidle_filter_path"] = source["data"]["nonidle_filter_path"]
    subset["data"]["pretrained_norm_stats"] = source["data"]["pretrained_norm_stats"]
    for key, value in source["model"].items():
        if key in {"transformer_path", "autoencoder_path", "text_encoder_path", "flux_source"}:
            subset["model"][key] = value
    assert subset == source


def test_release_defaults_are_unfiltered_and_fit_their_own_statistics(environment, monkeypatch):
    monkeypatch.delenv("PATCHWAM_DATA_ROOT", raising=False)
    monkeypatch.delenv("PATCHWAM_NONIDLE_FILTER", raising=False)
    monkeypatch.delenv("PATCHWAM_NORM_STATS", raising=False)
    config = read_configuration(CONFIG_DIR / "robotwin_vlm_5tasks.yaml")
    assert list(config.data.dataset_dirs) == FIVE_ROOTS
    # This release ships no row filter, and statistics are fitted from the
    # selected roots unless the environment supplies them.
    assert config.data.nonidle_filter_path is None
    assert config.data.pretrained_norm_stats is None
    monkeypatch.setenv("PATCHWAM_DATA_ROOT", "/tmp/other_parent")
    monkeypatch.setenv("PATCHWAM_NONIDLE_FILTER", "/tmp/filter.json")
    monkeypatch.setenv("PATCHWAM_NORM_STATS", "/tmp/stats.json")
    moved = read_configuration(CONFIG_DIR / "robotwin_vlm_5tasks.yaml")
    assert list(moved.data.dataset_dirs) == [f"/tmp/other_parent/{task}" for task in FIVE_TASKS]
    assert moved.data.nonidle_filter_path == "/tmp/filter.json"
    assert moved.data.pretrained_norm_stats == "/tmp/stats.json"


def test_selector_name_matches_the_dataset_parameter(environment):
    parameters = inspect.signature(EpisodeDataset.__init__).parameters
    assert "task_selection" in parameters
    assert parameters["task_selection"].default is None


def test_config_roots_reach_the_dataset(environment, tmp_path):
    from test_data_task_selection import make_root, write_stats

    root = make_root(tmp_path, "hanging_mug", [(0, 0, "hang the mug"),
                                               (1, 0, "hang the mug")], length=20,
                     action_width=14, state_width=14,
                     image_columns=("observation.images.cam_high",
                                    "observation.images.cam_left_wrist",
                                    "observation.images.cam_right_wrist"))
    config = read_configuration(CONFIG_DIR / "robotwin_vlm_5tasks.yaml", [
        f"data.dataset_dirs=[{root}]",
        f"data.pretrained_norm_stats={write_stats(tmp_path, action_width=14, state_width=14)}",
        "data.nonidle_filter_path=null",
        "data.val_set_proportion=0",
    ])
    dataset = construct(config.data)
    assert [episode.index for episode in dataset.episodes] == [0, 1]
    # Listing roots is the whole filter, so no task selection is recorded.
    assert dataset.task_selection is None
