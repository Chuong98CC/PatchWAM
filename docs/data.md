# Data preparation

[Overview](../README.md) · [Model assets](models.md) · [Training recipes](training_recipes.md) · [Evaluation](evaluation.md)

Use local LeRobot v2 datasets. Configure dataset roots explicitly; this repository
does not download or include research data. Preserve each benchmark's action
semantics, action horizon, normalization statistics, non-idle window filter, and
camera arrangement when reproducing an existing run. A changed data contract is
a separate experiment, even when the optimizer settings are identical.

## LeRobot v2 layout

Each entry in `data.dataset_dirs` must be a dataset root with its own `meta/info.json`.
A typical root contains:

```text
dataset_root/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   └── tasks.jsonl
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── episode_000001.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.cam_high/
        │   └── episode_000000.mp4
        ├── observation.images.cam_left_wrist/
        │   └── episode_000000.mp4
        └── observation.images.cam_right_wrist/
            └── episode_000000.mp4
```

Camera names vary by recipe. Images may also be embedded in parquet records. Custom
data/video path templates come from `meta/info.json`. LeRobot v3 merged storage
requires a separate reader and is currently unsupported.

The default feature columns are `action`, `observation.state`, and
`observation.images.<camera_name>`. Configure `data.shape_meta` to match the actual
fields, camera order, dimensions, and resize shapes. A field's `lerobot_key` can
override the default column name. The reader raises on unsupported formats or
missing action fields.

Set the root and normalization-statistics paths:

```bash
export PATCHWAM_DATA_ROOT=/path/to/lerobot/dataset_root
export PATCHWAM_NORM_STATS=/path/to/statistics.json
# Required by the RobotWin and C2R configurations:
export PATCHWAM_NONIDLE_FILTER=/path/to/nonidle_ranges.json
```

For several roots, override the list explicitly:

```bash
accelerate launch --multi_gpu --num_machines 1 --num_processes 8 --mixed_precision bf16 \
  -m patchwam.cli train --config configs/robocasa.yaml \
  'data.dataset_dirs=[/data/task_a,/data/task_b]' \
  training.accumulation=2 \
  training.output_dir=runs/robocasa-multi-root
```

All roots in one dataset must share FPS. The reader validates timestamps against
that FPS and matches decoded frames with a default tolerance of `1e-4` seconds.
`data.lerobot_tolerance_s` changes the tolerance explicitly. The reader does not
resample datasets to a different control frequency.

## Windows, normalization, and filters

The supplied recipes use `num_frames: 17`, `endpoint_frames_only: true`, and an
action/video frequency ratio of 1. A sample contains current/future images at
`t` and `t+16`, and 16 actions at `t ... t+15`. Episode-end windows repeat the
last available observation and mark padded action steps separately.

RoboTwin uses 14 action/state dimensions and a compact three-camera image.
LIBERO uses 7 action dimensions, 8 state dimensions, and a horizontal two-camera
image. The supplied RoboCasa configuration uses 12 action and 16 state dimensions;
other dataset versions can use different interfaces.

Use normalization statistics from the exact dataset and action contract being
trained. RobotWin and the supplied RoboCasa configuration use z-score scaling;
LIBERO uses min/max scaling. The data processor also supports quantile scaling,
stepwise action statistics, and per-embodiment tables when configured explicitly.
Preserve the same transforms when decoding actions at inference.

Setting `data.pretrained_norm_stats=null` computes statistics from the selected
training split. To keep those statistics for later inference or validation,
export them explicitly:

```python
from patchwam.configuration import construct, read_configuration
from patchwam.data.scaling import write_statistics

config = read_configuration(
    "configs/robotwin.yaml", ["data.pretrained_norm_stats=null"]
)
dataset = construct(config.data)
write_statistics(dataset.statistics, "statistics.json")
```

This assumes the other environment paths are configured. For reproduction, use
the original statistics rather than silently fitting a new normalization contract.

RobotWin configurations accept episode-indexed non-idle ranges. Ranges select
episode rows and use an exclusive end index; windows are assembled from this
filtered row sequence and may cross between retained intervals. Missing episode
entries keep all rows for that episode, so check that the filter belongs to the
supplied roots. For multiple roots with separate filters,
`data.nonidle_filter_path` can be a mapping from each resolved absolute dataset-root
path to its corresponding filter file.
The C2R configuration additionally keeps the first 50 episodes of each 550-episode
block and applies temporally consistent appearance augmentation with probability 0.8.

## Task selection

`data.task_selection` narrows a run to exact task names. Leave it unset to keep
the full dataset. Task *names*, not local indices, are the identity, so the
selector is safe for a merged root whose `meta/tasks.jsonl` numbering differs
from another root:

```yaml
data:
  task_selection:
    tasks:
    - turn_switch
    - rotate_qrcode
    - blocks_ranking_size
    - move_stapler_pad
    - hanging_mug
```

The selection resolves before the train/validation split and before statistics
or sample indexes, and it composes with `episode_index_filter` and the non-idle
row filter. Each episode resolves its task from `meta/episodes.jsonl` (`tasks`,
`task`, or a root-local `task_index`) and falls back to the parquet
`task_index` column when the metadata carries none. A requested name that
appears in no root, an episode without a resolvable identity, and an episode
mixing several task identities all raise instead of silently training on the
full data. `configs/robotwin_vlm_5tasks.yaml` uses this for the five-task
ablation: it copies `configs/robotwin_vlm.yaml` unchanged apart from the
selector and `training.output_dir`, so model, conditioning, augmentation,
non-idle filtering, horizon, dimensions, optimizer, epochs, and batch settings
are preserved.

The subset still needs statistics that describe the same five tasks. Point
`PATCHWAM_NORM_STATS` at matching statistics, or fit them from the selected
split with `data.pretrained_norm_stats=null`; the normalization method itself is
unchanged.

## Optional text caches

The FLUX.2 policy can consume cached Qwen3 features. Set these data options:

```yaml
qwen_text_cache_dir: /path/to/text_cache
qwen_text_cache_format: qwen3_flux2
qwen_context_len: 128
```

Cache files use the SHA256 hash of the final formatted prompt:
`<prompt_hash>.qwen3_flux2_len128.pt`. Each payload contains `text_hidden_states`
with shape `[128, text_feature_width]` and `text_attention_mask` with shape `[128]`.
The reader supplies native `text_tokens` and `text_valid` to the policy. Cached
features must match the encoder, prompt template, and context length used in training.
Incompatible cache formats are rejected by the FLUX.2 policy.

## VLM and causal histories

The optional RoboDojo recipe uses separate camera views, causal past-frame slots,
unaugmented VL inputs, and per-row subtask sentence labels. Slot selection and the
episode-local online buffer are independent Apache-2.0 implementations. See
[conditioning and training options](conditioning_and_training.md) for the input
contract and [evaluation](evaluation.md) for online history reset.

## Dataset fingerprint and provenance

The dataset fingerprint includes metadata, normalization, selected rows, and file
size/modification-time records for episode tables, external camera videos, and active
text caches. It does not hash full payloads or inspect external image paths embedded
inside parquet records. Keep those data assets immutable during a run.

See [data-processing provenance](../src/patchwam/data/PROVENANCE.json) for the adapted
functions and their source licenses, along with [NOTICE](../NOTICE) and the retained
[MIT data-processing license](../licenses/ImageWAM-MIT.txt).
