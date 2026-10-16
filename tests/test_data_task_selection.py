"""Exact task-name selection over synthetic local LeRobot v2 roots."""

import json
import inspect
import subprocess
import sys
import types
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from patchwam.data import EpisodeDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Revision the selector work started from; the last one without a selector.
BASELINE_REVISION = "dea19d21182a26a8ee16efc2c4125ee7aecf44fb"

SHAPE_META = {
    "images": [{"key": "front", "raw_shape": [3, 8, 8], "shape": [3, 8, 8]}],
    "action": [{"key": "default", "raw_shape": 2, "shape": 2}],
    "state": [{"key": "default", "raw_shape": 3, "shape": 3}],
}

FIVE_TASKS = ["turn_switch", "rotate_qrcode", "blocks_ranking_size",
              "move_stapler_pad", "hanging_mug"]
EXCLUDED_TASK = "place_object"


def legacy_episode_dataset():
    """Load the pre-selector ``EpisodeDataset`` from the baseline revision.

    Comparing live fingerprints against the recorded baseline source keeps the
    legacy-compatibility check meaningful without freezing a hash. Skipped when
    the revision is unavailable or already carries the selector.
    """
    try:
        source = subprocess.run(
            ["git", "show", f"{BASELINE_REVISION}:src/patchwam/data/episodes.py"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"baseline revision unavailable: {error}")
    if "task_selection" in source:
        pytest.skip("baseline revision already carries the selector")
    module = types.ModuleType("patchwam.data.episodes_baseline")
    module.__package__, module.__file__ = "patchwam.data", "<baseline>"
    sys.modules[module.__name__] = module
    exec(compile(source, "<baseline>", "exec"), module.__dict__)
    return module.EpisodeDataset


def _frames(length):
    frames = []
    for index in range(length):
        buffer = BytesIO()
        Image.fromarray(np.full((8, 8, 3), index * 10, dtype=np.uint8)).save(buffer, format="PNG")
        frames.append({"bytes": buffer.getvalue(), "path": None})
    return frames


def make_root(tmp_path, name, episodes, *, length=12, fps=20, metadata_tasks=True,
              episode_tasks=None, parquet_task_index=True, action_width=2, state_width=3,
              image_columns=("observation.images.front",)):
    """Write a minimal LeRobot v2 root.

    ``episodes`` is a list of ``(episode_index, task_index, task_name)``.
    ``episode_tasks`` overrides what one episode declares in episodes.jsonl.
    """
    root = tmp_path / name
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {"fps": fps, "total_episodes": len(episodes), "chunks_size": 1000,
            "codebase_version": "v2.1",
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}
    (root / "meta/info.json").write_text(json.dumps(info))
    table_rows = {}
    for _, task_index, task in episodes:
        table_rows[task_index] = task
    (root / "meta/tasks.jsonl").write_text("".join(
        json.dumps({"task_index": index, "task": task}) + "\n"
        for index, task in sorted(table_rows.items())))
    entries = []
    for episode_index, _, task in episodes:
        entry = {"episode_index": episode_index, "length": length}
        if metadata_tasks:
            entry["tasks"] = (episode_tasks or {}).get(episode_index, [task])
        entries.append(json.dumps(entry))
    (root / "meta/episodes.jsonl").write_text("\n".join(entries) + "\n")
    for episode_index, task_index, _ in episodes:
        table = {
            "action": [[float(step + offset) for offset in range(action_width)]
                       for step in range(length)],
            "observation.state": [[float(step + offset) for offset in range(state_width)]
                                  for step in range(length)],
            "timestamp": [step / fps for step in range(length)],
        }
        for column in image_columns:
            table[column] = _frames(length)
        if parquet_task_index:
            table["task_index"] = [task_index] * length
        pq.write_table(pa.table(table),
                       root / "data/chunk-000" / f"episode_{episode_index:06d}.parquet")
    return root


def write_stats(tmp_path, name="stats.json", action_width=2, state_width=3):
    statistics = {field: {"default": {"global_min": [0] * width, "global_max": [20] * width,
                                      "global_q01": [0] * width, "global_q99": [20] * width,
                                      "global_mean": [10] * width, "global_std": [5] * width}}
                  for field, width in (("action", action_width), ("state", state_width))}
    path = tmp_path / name
    path.write_text(json.dumps(statistics))
    return path


def open_dataset(roots, stats, **kwargs):
    return EpisodeDataset(list(roots), SHAPE_META, pretrained_norm_stats=stats,
                          video_size=(8, 8), **kwargs)


def five_task_episodes():
    """One merged root: five requested tasks plus one excluded task."""
    return [(0, 0, "turn_switch"), (1, 0, "turn_switch"),
            (2, 1, "rotate_qrcode"), (3, 2, "blocks_ranking_size"),
            (4, 3, "move_stapler_pad"), (5, 4, "hanging_mug"),
            (6, 5, EXCLUDED_TASK)]


def test_selector_keeps_only_requested_tasks(tmp_path):
    root = make_root(tmp_path, "merged", five_task_episodes())
    dataset = open_dataset([root], write_stats(tmp_path), task_selection={"tasks": FIVE_TASKS})
    assert [episode.index for episode in dataset.episodes] == [0, 1, 2, 3, 4, 5]
    assert len(dataset) == 6 * 12
    assert dataset.task_selection == {
        "tasks": FIVE_TASKS, "selected_episodes": {str(root.resolve()): [0, 1, 2, 3, 4, 5]}}


def test_selector_absent_keeps_legacy_behavior(tmp_path):
    root = make_root(tmp_path, "merged", five_task_episodes())
    dataset = open_dataset([root], write_stats(tmp_path))
    assert [episode.index for episode in dataset.episodes] == list(range(7))
    assert len(dataset) == 7 * 12
    assert dataset.task_selection is None


def test_task_filter_runs_before_the_train_validation_split(tmp_path):
    root = make_root(tmp_path, "merged", five_task_episodes())
    stats = write_stats(tmp_path)
    selection = {"tasks": FIVE_TASKS}
    train = open_dataset([root], stats, task_selection=selection, val_set_proportion=0.5)
    validation = open_dataset([root], stats, task_selection=selection,
                              val_set_proportion=0.5, is_training_set=False)
    order = [0, 1, 2, 3, 4, 5]
    np.random.default_rng(42).shuffle(order)
    boundary = int(len(order) * 0.5)
    train_indices = [episode.index for episode in train.episodes]
    val_indices = [episode.index for episode in validation.episodes]
    assert train_indices == order[:boundary]
    assert val_indices == order[boundary:]
    assert set(train_indices) | set(val_indices) == set(range(6))


def test_task_names_are_the_identity_across_roots(tmp_path):
    first = make_root(tmp_path, "root_a", [(0, 0, "pick"), (1, 0, "pick"), (2, 1, "place")])
    second = make_root(tmp_path, "root_b", [(0, 0, "wipe"), (1, 0, "wipe")])
    dataset = open_dataset([first, second], write_stats(tmp_path),
                           task_selection={"tasks": ["wipe", "place"]})
    assert dataset.task_selection["selected_episodes"] == {str(first.resolve()): [2],
                                                           str(second.resolve()): [0, 1]}
    assert [episode.root for episode in dataset.episodes] == [first.resolve(), second.resolve(),
                                                              second.resolve()]


def test_missing_episode_metadata_falls_back_to_parquet_task_index(tmp_path):
    root = make_root(tmp_path, "merged", five_task_episodes(), metadata_tasks=False)
    dataset = open_dataset([root], write_stats(tmp_path), task_selection={"tasks": FIVE_TASKS})
    assert [episode.index for episode in dataset.episodes] == [0, 1, 2, 3, 4, 5]


def test_metadata_indices_resolve_through_the_root_table(tmp_path):
    episodes = [(0, 0, "turn_switch"), (1, 3, "move_stapler_pad"), (2, 5, EXCLUDED_TASK)]
    root = make_root(tmp_path, "merged", episodes, episode_tasks={0: 0, 1: 3, 2: 5})
    selection = {"tasks": ["turn_switch", "move_stapler_pad"]}
    dataset = open_dataset([root], write_stats(tmp_path), task_selection=selection)
    assert [episode.index for episode in dataset.episodes] == [0, 1]


@pytest.mark.parametrize("specification,error,message", [
    ({"tasks": ["turn_switch", "absent_task"]}, ValueError, "absent from the dataset roots"),
    ({"tasks": []}, ValueError, "at least one task name"),
    ({"tasks": ["turn_switch", "turn_switch"]}, ValueError, "Duplicate task"),
    ({"tasks": FIVE_TASKS, "num_tasks": 2}, ValueError, "Unknown task selection options"),
    ({}, ValueError, "requires an explicit 'tasks' list"),
    ({"tasks": ["turn_switch"], "expected_episodes": 2}, ValueError,
     "only applies to an episode_ranges selection"),
    (["turn_switch"], TypeError, "must be a mapping"),
    ({"tasks": "turn_switch"}, TypeError, "must be a list of exact task names"),
    ({"tasks": [1]}, TypeError, "non-empty strings"),
])
def test_invalid_selectors_fail_loudly(tmp_path, specification, error, message):
    root = make_root(tmp_path, "merged", five_task_episodes())
    with pytest.raises(error, match=message):
        open_dataset([root], write_stats(tmp_path), task_selection=specification)


def test_ambiguous_episodes_are_rejected(tmp_path):
    episodes = [(0, 0, "turn_switch"), (1, 1, "rotate_qrcode"), (2, 5, EXCLUDED_TASK)]
    selection = {"tasks": ["turn_switch", "rotate_qrcode"]}
    mixed = make_root(tmp_path, "mixed", episodes,
                      episode_tasks={1: ["rotate_qrcode", EXCLUDED_TASK]})
    with pytest.raises(ValueError, match="mixes task identities"):
        open_dataset([mixed], write_stats(tmp_path), task_selection=selection)
    unknown = make_root(tmp_path, "unknown", episodes, episode_tasks={1: ["absent_task"]})
    with pytest.raises(ValueError, match="absent from the root's meta/tasks.jsonl"):
        open_dataset([unknown], write_stats(tmp_path), task_selection=selection)
    bare = make_root(tmp_path, "bare", episodes, metadata_tasks=False, parquet_task_index=False)
    with pytest.raises(ValueError, match="has no task identity"):
        open_dataset([bare], write_stats(tmp_path), task_selection=selection)


def test_task_selection_composes_with_nonidle_rows_and_statistics(tmp_path):
    root = make_root(tmp_path, "merged", five_task_episodes())
    nonidle = tmp_path / "nonidle.json"
    nonidle.write_text(json.dumps(
        {"episodes": {str(index): [[0, 2], [6, 10]] for index in range(7)}}))
    dataset = open_dataset([root], None, task_selection={"tasks": FIVE_TASKS}, num_frames=5,
                           nonidle_filter_path=nonidle, is_training_set=True)
    assert len(dataset) == 6 * 6
    assert dataset.statistics["num_episodes"] == 6
    assert dataset.statistics["num_transition"] == 36
    full = open_dataset([root], None, num_frames=5, nonidle_filter_path=nonidle)
    assert full.statistics["num_episodes"] == 7


def test_unselected_fingerprint_matches_the_pre_selector_baseline(tmp_path):
    baseline = legacy_episode_dataset()
    root = make_root(tmp_path, "merged", five_task_episodes())
    stats = write_stats(tmp_path)
    current = EpisodeDataset([root], SHAPE_META, pretrained_norm_stats=stats, video_size=(8, 8))
    legacy = baseline([root], SHAPE_META, pretrained_norm_stats=stats, video_size=(8, 8))
    assert current.task_selection is None
    assert current.fingerprint == legacy.fingerprint


def test_selector_is_appended_after_the_existing_optionals():
    baseline = legacy_episode_dataset()
    previous = list(inspect.signature(baseline.__init__).parameters)
    current = list(inspect.signature(EpisodeDataset.__init__).parameters)
    assert current[:len(previous)] == previous
    assert current[len(previous):] == ["task_selection"]


def test_requested_task_without_candidate_episodes_is_rejected(tmp_path):
    root = make_root(tmp_path, "merged", [(0, 0, "turn_switch"), (1, 0, "turn_switch")])
    tasks_path = root / "meta/tasks.jsonl"
    tasks_path.write_text(tasks_path.read_text()
                          + json.dumps({"task_index": 1, "task": "hanging_mug"}) + "\n")
    stats = write_stats(tmp_path)
    with pytest.raises(ValueError, match="no candidate episodes"):
        open_dataset([root], stats, task_selection={"tasks": ["turn_switch", "hanging_mug"]})
    kept = open_dataset([root], stats, task_selection={"tasks": ["turn_switch"]})
    assert [episode.index for episode in kept.episodes] == [0, 1]


def test_requested_task_removed_by_the_episode_filter_is_rejected(tmp_path):
    root = make_root(tmp_path, "merged", [(0, 0, "turn_switch"), (1, 0, "turn_switch"),
                                          (2, 1, "rotate_qrcode")])
    periodic = {"mode": "first_k_per_period", "period": 2, "keep_first": 1, "offset": 1}
    with pytest.raises(ValueError, match="no candidate episodes"):
        open_dataset([root], write_stats(tmp_path), episode_index_filter=periodic,
                     task_selection={"tasks": ["turn_switch", "rotate_qrcode"]})


def test_each_task_need_not_survive_the_train_validation_split(tmp_path):
    root = make_root(tmp_path, "merged", [(0, 0, "turn_switch"), (1, 1, "rotate_qrcode")])
    stats, selection = write_stats(tmp_path), {"tasks": ["turn_switch", "rotate_qrcode"]}
    train = open_dataset([root], stats, task_selection=selection, val_set_proportion=0.5)
    validation = open_dataset([root], stats, task_selection=selection, val_set_proportion=0.5,
                              is_training_set=False)
    names = {0: "turn_switch", 1: "rotate_qrcode"}
    train_names = {names[episode.index] for episode in train.episodes}
    val_names = {names[episode.index] for episode in validation.episodes}
    assert len(train.episodes) == len(validation.episodes) == 1
    assert train_names != val_names


def test_root_split_does_not_decide_candidate_presence(tmp_path):
    first = make_root(tmp_path, "root_a", [(0, 0, "turn_switch"), (1, 0, "turn_switch")])
    second = make_root(tmp_path, "root_b", [(0, 0, "rotate_qrcode")])
    stats = write_stats(tmp_path)
    selection = {"tasks": ["turn_switch", "rotate_qrcode"]}
    train = open_dataset([first, second], stats, task_selection=selection,
                         val_set_proportion=0.5, val_split_level="root")
    validation = open_dataset([first, second], stats, task_selection=selection,
                              val_set_proportion=0.5, val_split_level="root", is_training_set=False)
    train_roots = set(train.task_selection["selected_episodes"])
    val_roots = set(validation.task_selection["selected_episodes"])
    assert train.episodes and validation.episodes
    assert train_roots | val_roots == {str(first.resolve()), str(second.resolve())}
    assert not train_roots & val_roots


def test_episode_ranges_select_release_style_paraphrase_episodes(tmp_path):
    """Benchmark identity by episode range, not by the paraphrase text."""
    episodes = [(0, 0, "push the switch"), (1, 1, "rotate the qr code")]
    paraphrases = {0: [f"instruction variant {index}" for index in range(5)],
                   1: [f"other variant {index}" for index in range(3)]}
    root = make_root(tmp_path, "release", episodes, episode_tasks=paraphrases)
    selection = {"tasks": ["turn_switch"], "expected_episodes": 2,
                 "episode_ranges": {"turn_switch": [[0, 1]]}}
    dataset = open_dataset([root], write_stats(tmp_path), task_selection=selection)
    assert [episode.index for episode in dataset.episodes] == [0]
    assert dataset.task_selection["episode_ranges"] == {"turn_switch": [[0, 1]]}
    assert "push the switch" in dataset[0]["instruction"]


def test_episode_ranges_compose_with_the_split_and_nonidle_filter(tmp_path):
    episodes = [(0, 0, "first"), (1, 0, "first"), (2, 1, "second"), (3, 1, "second")]
    paraphrases = {index: [f"variant {index}.{variant}" for variant in range(4)]
                   for index in range(4)}
    root = make_root(tmp_path, "release", episodes, episode_tasks=paraphrases)
    nonidle = tmp_path / "nonidle.json"
    nonidle.write_text(json.dumps({"episodes": {str(index): [[0, 2], [6, 10]]
                                                for index in range(4)}}))
    selection = {"tasks": ["first", "second"], "expected_episodes": 4,
                 "episode_ranges": {"first": [[0, 2]], "second": [[2, 4]]}}
    dataset = open_dataset([root], None, task_selection=selection, num_frames=5,
                           nonidle_filter_path=nonidle, is_training_set=True)
    assert dataset.statistics["num_episodes"] == 4
    assert dataset.statistics["num_transition"] == 4 * 6
    train = open_dataset([root], write_stats(tmp_path), task_selection=selection,
                         val_set_proportion=0.5)
    assert len(train.episodes) == 2


@pytest.mark.parametrize("selection,message", [
    ({"tasks": ["turn_switch"], "episode_ranges": {"turn_switch": [[0, 9]], "extra": [[0, 1]]}},
     "must name exactly the requested tasks"),
    ({"tasks": ["first", "second"], "episode_ranges": {"first": [[0, 1]]}},
     "must name exactly the requested tasks"),
    ({"tasks": ["turn_switch"], "episode_ranges": {"turn_switch": []}},
     "nonempty list"),
    ({"tasks": ["turn_switch"], "episode_ranges": {"turn_switch": [[1, 1]]}},
     "0 <= start < end"),
    ({"tasks": ["turn_switch"], "episode_ranges": {"turn_switch": [[0, 1], [0, 2]]}},
     "overlaps another mapped range"),
    ({"tasks": ["turn_switch"], "episode_ranges": {"turn_switch": ["x"]}},
     "two integers"),
    ({"tasks": ["turn_switch"], "expected_episodes": 3, "episode_ranges": {"turn_switch": [[0, 1]]}},
     "declares 2 episodes but the selection expects 3"),
    ({"tasks": ["turn_switch"], "episode_ranges": {"turn_switch": [[0, 5]]}},
     "exceeds the 2 episodes declared"),
])
def test_invalid_episode_ranges_fail_loudly(tmp_path, selection, message):
    root = make_root(tmp_path, "release", [(0, 0, "first"), (1, 0, "first")])
    with pytest.raises(ValueError, match=message):
        open_dataset([root], write_stats(tmp_path), task_selection=selection)


def test_episode_ranges_require_a_single_root(tmp_path):
    first = make_root(tmp_path, "root_a", [(0, 0, "first")])
    second = make_root(tmp_path, "root_b", [(0, 0, "second")])
    selection = {"tasks": ["first", "second"],
                 "episode_ranges": {"first": [[0, 1]], "second": [[1, 2]]}}
    with pytest.raises(ValueError, match="exactly one dataset root"):
        open_dataset([first, second], write_stats(tmp_path), task_selection=selection)
