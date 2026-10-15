"""Exact task-name episode selection for local LeRobot v2 roots.

LeRobot task indices are root-local, so a task *name* is the only identity that
compares safely across roots: index 0 may mean different tasks in different
roots. The selector is opt-in. Without ``task_selection`` the dataset keeps its
previous behavior; with it, matching episodes resolve before the
train/validation split and before statistics or sample-index generation, so a
narrowed run never silently falls back to the full dataset. Candidates are
resolved across every configured root before any train/validation partition,
and each requested name must own at least one retained episode.

Only ``{"tasks": [name, ...]}`` with exact names is supported. That bounded
shape matches the FluxVLA ``datasets.task_selection`` option without adding
seeded sampling or episode caps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ["build_task_table", "episode_task_names", "normalize_task_selection",
           "resolve_episode_tasks"]


def normalize_task_selection(task_selection):
    """Validate the opt-in selector and return the requested exact names.

    Returns ``None`` when no selector is configured, which keeps the legacy
    unfiltered behavior available to every existing recipe.
    """
    if task_selection is None:
        return None
    if not isinstance(task_selection, Mapping):
        raise TypeError(
            "task_selection must be a mapping with a 'tasks' list, received "
            f"{type(task_selection).__name__}"
        )
    unknown = sorted(str(key) for key in set(task_selection) - {"tasks"})
    if unknown:
        raise ValueError(f"Unknown task selection options: {unknown}")
    if "tasks" not in task_selection:
        raise ValueError("Task selection requires an explicit 'tasks' list of exact task names")
    requested = task_selection["tasks"]
    if isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence):
        raise TypeError(
            "Task selection 'tasks' must be a list of exact task names, received "
            f"{type(requested).__name__}"
        )
    names = []
    for item in requested:
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"Task names must be non-empty strings, received {item!r}")
        if item in names:
            raise ValueError(f"Duplicate task in selection: {item!r}")
        names.append(item)
    if not names:
        raise ValueError("Task selection requires at least one task name")
    return tuple(names)


def build_task_table(rows):
    """Root-local ``task_index -> task name`` table from ``meta/tasks.jsonl`` rows."""
    table = {}
    for row in rows:
        index = int(row["task_index"])
        name = str(row["task"])
        if index in table and table[index] != name:
            duplicate = sorted({table[index], name})
            raise ValueError(f"Ambiguous task names for index {index}: {duplicate}")
        table[index] = name
    return table


def _task_name(value, table, known):
    """Resolve one metadata value to an exact task name."""
    if isinstance(value, str):
        name = value.strip()
        if name in known:
            return name
        if name.lstrip("-").isdigit() and int(name) in table:
            return table[int(name)]
        raise ValueError(f"Episode task {value!r} is absent from the root's meta/tasks.jsonl")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Episode task must be a name or a root-local index, received {value!r}")
    if value not in table:
        raise ValueError(f"Episode task index {value} is absent from the root's meta/tasks.jsonl")
    return table[value]


def episode_task_names(entry, table, parquet_task_indices=None, known_names=None):
    """Exact task names declared by one episode.

    Episode metadata is authoritative when it carries task identities, either as
    task names (``tasks``/``task``) or as root-local indices (``task_index``).
    Roots whose metadata omits them fall back to the parquet ``task_index``
    values, which the caller reads and passes as ``parquet_task_indices``.
    Returns an empty set when neither source resolves an identity.
    """
    known = set(table.values()) if known_names is None else known_names
    values = []
    if entry is not None:
        raw = entry.get("tasks")
        if raw is None:
            single = entry.get("task")
            raw = [] if single in (None, "") else [single]
        elif not isinstance(raw, (list, tuple)):
            raw = [raw]
        values = list(raw)
        if not values and entry.get("task_index") is not None:
            values = [entry["task_index"]]
    if values:
        return {_task_name(value, table, known) for value in values}
    if parquet_task_indices is None:
        return set()
    return {_task_name(value, table, known) for value in parquet_task_indices}


def resolve_episode_tasks(indices, entries, table, root, parquet_task_indices=None):
    """Map each episode index to its single exact task name.

    Args:
        indices: Episode indices to resolve.
        entries: ``episode_index -> episodes.jsonl row`` for this root.
        table: Root-local ``task_index -> task name`` table.
        root: Dataset root, used in error messages.
        parquet_task_indices: Optional ``index -> task_index values`` callback
            used only when the episode metadata carries no task identity.

    Returns:
        ``episode_index -> task name`` for every requested index.

    Raises:
        ValueError: when an episode cannot be resolved to exactly one task.
    """
    known = set(table.values())
    resolved = {}
    for index in indices:
        names = episode_task_names(entries.get(index), table, known_names=known)
        if not names and parquet_task_indices is not None:
            values = parquet_task_indices(index)
            if values:
                names = episode_task_names(
                    None, table, parquet_task_indices=values, known_names=known)
        if not names:
            raise ValueError(
                f"Episode {index} in {root} has no task identity: meta/episodes.jsonl carries "
                "no task and the parquet has no task_index column"
            )
        if len(names) > 1:
            raise ValueError(
                f"Episode {index} in {root} mixes task identities {sorted(names)}; a "
                "task-selected dataset needs exactly one task per episode"
            )
        resolved[index] = next(iter(names))
    return resolved
