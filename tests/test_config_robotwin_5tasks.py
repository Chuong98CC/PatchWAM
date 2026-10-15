"""Config units for the five-task RobotWin recipe."""

import inspect
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from patchwam.configuration import construct, read_configuration
from patchwam.data import EpisodeDataset

FIVE_TASKS = ["turn_switch", "rotate_qrcode", "blocks_ranking_size",
              "move_stapler_pad", "hanging_mug"]
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
ENVIRONMENT = {
    "PATCHWAM_TRANSFORMER": "/assets/transformer.safetensors",
    "PATCHWAM_AUTOENCODER": "/assets/autoencoder.safetensors",
    "PATCHWAM_VLM": "/assets/text_encoder",
    "FLUX2_SRC": "/assets/flux2",
    "PATCHWAM_DATA_ROOT": "/data/robotwin",
    "PATCHWAM_NORM_STATS": "/data/robotwin/statistics.json",
    "PATCHWAM_NONIDLE_FILTER": "/data/robotwin/nonidle.json",
}


@pytest.fixture
def environment(monkeypatch):
    for name, value in ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    return ENVIRONMENT


def test_five_task_config_is_the_source_recipe_plus_the_selector(environment):
    source = OmegaConf.to_container(
        read_configuration(CONFIG_DIR / "robotwin_vlm.yaml"), resolve=True)
    subset = OmegaConf.to_container(
        read_configuration(CONFIG_DIR / "robotwin_vlm_5tasks.yaml"), resolve=True)
    assert subset["data"]["task_selection"] == {"tasks": FIVE_TASKS}
    assert subset["training"]["output_dir"] == "runs/robotwin_vlm_5tasks"
    subset["data"].pop("task_selection")
    subset["training"]["output_dir"] = source["training"]["output_dir"]
    assert subset == source


def test_selector_name_matches_the_dataset_parameter(environment):
    parameters = inspect.signature(EpisodeDataset.__init__).parameters
    assert "task_selection" in parameters
    assert parameters["task_selection"].default is None


def test_config_selector_reaches_the_dataset(environment, tmp_path):
    from test_data_task_selection import five_task_episodes, make_root, write_stats

    root = make_root(
        tmp_path, "merged", five_task_episodes(), action_width=14, state_width=14,
        image_columns=("observation.images.cam_high", "observation.images.cam_left_wrist",
                       "observation.images.cam_right_wrist"),
    )
    config = read_configuration(CONFIG_DIR / "robotwin_vlm_5tasks.yaml", [
        f"data.dataset_dirs=[{root}]",
        f"data.pretrained_norm_stats={write_stats(tmp_path, action_width=14, state_width=14)}",
        "data.nonidle_filter_path=null",
        "data.val_set_proportion=0",
    ])
    dataset = construct(config.data)
    assert [episode.index for episode in dataset.episodes] == [0, 1, 2, 3, 4, 5]
    assert list(dataset.task_selection["tasks"]) == FIVE_TASKS
