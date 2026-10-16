# Optional DiT adapter finetuning

[Overview](../README.md) · [Model assets](models.md) · [Training](training.md) · [Conditioning and training options](conditioning_and_training.md)

The default recipe fully finetunes the FLUX.2 denoising transformer. The optional
DiT adapter mode keeps that recipe and adds [Hugging Face PEFT](https://huggingface.co/docs/peft)
adapters to individual linear projections. Only the adapted projection weights are
frozen; every other pretrained transformer parameter keeps training exactly as it
does without adapters. The default adapter is LoRA over the block attention and
feed-forward projections, and `dit_adapter_config` selects other PEFT families and
targets.

## Adapter families

| `peft_type` | Status | Notes |
| --- | --- | --- |
| `LORA` | Default | `r`, `lora_alpha`, `lora_dropout`; zero-initialized up-projection so a fresh adapter is the identity |
| `IA3` | Supported | Ones-initialized rescaling vectors; `feedforward_modules` follow the official IA3 convention |

Prompt-learning types (`PROMPT_TUNING`, `P_TUNING`, `PREFIX_TUNING`), DoRA,
`modules_to_save`, adapter bias, and parameter-level or layer-index targeting are
rejected with an actionable error naming the option rather than accepted and
ignored. The
masked manual execution in `OfficialFluxDenoiser` calls the official module
attributes directly, so adapters are injected with PEFT's low-level
[`inject_adapter_in_model`](https://huggingface.co/docs/peft/developer_guides/low_level_api)
instead of `get_peft_model`. That keeps the official module paths and the policy
state dictionary intact: no `PeftModel` wrapper is retained, and no base-model
parameters are duplicated.

## Freeze contract

| Component | Adapter mode | Full mode (default) |
| --- | --- | --- |
| Adapted projection weights (`<projection>.base_layer.weight`) | Frozen | Fully trained |
| Adapter tensors (`lora_A`/`lora_B` for LoRA, `ia3_l` for IA3) | Trained | Not present |
| `img_in`, `txt_in`, `time_in`, stream modulations, `final_layer`, attention QK-norm scales, every other block weight | Fully trained | Fully trained |
| `policy.state_projection` | Fully trained | Fully trained |
| `policy.representation_projection` (Self-Flow variants) | Fully trained | Fully trained |
| Autoencoder | Frozen, evaluation mode | Frozen, evaluation mode |
| Text encoder, including VL adapters | Frozen, evaluation mode | Frozen, or trained when `vl_trainable_adapters=true` |

Enabling an adapter with `vl_trainable_adapters=true` fails at construction instead
of silently ignoring one of the two requests:

```text
ValueError: DiT adapters freeze the complete text encoder, including VL adapters;
set vl_trainable_adapters=false or disable the DiT adapter
```

The adapted projection weights are frozen and the adapter tensors are trainable
after injection, and the start-of-training hook re-asserts exactly that. PEFT
freezes every other parameter during injection; the implementation restores the
pre-injection `requires_grad` state of each original parameter, so the rest of the
pretrained transformer keeps its full-mode training contract. Injection also
preserves each projection's dtype and device, including `bfloat16`.

## Enabling it

The mode is a model option. It composes with an existing recipe through dotted
overrides, so no benchmark configuration has to change:

```bash
accelerate launch --multi_gpu --num_machines 1 --num_processes 8 --mixed_precision bf16 \
  -m patchwam.cli train --config configs/robotwin_vlm.yaml \
  model.dit_lora=true model.dit_lora_rank=64 model.dit_lora_alpha=128 model.dit_lora_dropout=0.05 \
  training.output_dir=runs/robotwin-vlm-dit-lora
```

| Option | Default | Meaning |
| --- | --- | --- |
| `model.dit_lora` | `false` | Adapt the default block projection inventory with LoRA and train it alongside the rest of the DiT |
| `model.dit_lora_rank` | `64` | LoRA bottleneck width; a positive integer |
| `model.dit_lora_alpha` | `128` | LoRA residual multiplier `alpha / rank`; finite and positive |
| `model.dit_lora_dropout` | `0.05` | Dropout on the LoRA input path, in `[0,1)` |
| `model.dit_adapter_config` | `null` | Explicit PEFT adapter mapping; see below |

`OfficialFluxDenoiser` accepts the same keyword arguments directly, including
`dit_adapter_config`, which is the API the CPU layer-contract tests use. Invalid
settings raise `ValueError` naming the offending argument, for example
`dit_lora_rank must be a positive integer`.

### Choosing the adapter explicitly

`model.dit_adapter_config` (a mapping passed to
`FluxAssetPolicy.from_local_assets`) enables PEFT on its own, without
`dit_lora`, and takes standard PEFT keys. `peft_type` defaults to `LORA`:

```yaml
model:
  _target_: patchwam.models.FluxAssetPolicy.from_local_assets
  dit_adapter_config:
    peft_type: IA3
```

```yaml
model:
  _target_: patchwam.models.FluxAssetPolicy.from_local_assets
  dit_adapter_config:
    peft_type: LORA
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: [txt_in, img_in, time_in.in_layer]
```

The mapping owns every field it declares. `target_modules` is optional: when it
is omitted the default block projection inventory below is adapted, and when it is
given, each entry is an exact dotted module path or a dotted-path suffix such as
`img_attn.qkv` or `img_mlp.0`. Every entry must resolve to an individual
`nn.Linear` of the official transformer; containers such as `double_blocks` or
`time_in`, names that match nothing, and non-linear matches are rejected with the
available names listed. That makes any single projection of the transformer
adaptable, including `txt_in`, `img_in`, the `time_in` and modulation layers, and
`final_layer`. Encoders stay out of scope.

Precedence between the two sources is explicit and validated: `dit_lora` plus the
legacy `dit_lora_rank`/`dit_lora_alpha`/`dit_lora_dropout` flags build a LORA
config when no mapping is given, and a non-default flag that contradicts a value
the mapping declares raises `ValueError` instead of being ignored. IA3 does not use
`r`/`lora_alpha`/`lora_dropout`, so passing a non-default one of those flags
together with an IA3 mapping is rejected too.

## Initializing from a pretrained checkpoint

`--initialize` accepts the upstream release `.pt` file as well as native
`.safetensors` weights. The 149 pretrained transformer tensors initialize the
model, the adapted projections receive theirs as the frozen `base_layer` weights,
the `proprio_encoder` tensors initialize `policy.state_projection`, and the 80
adapter pairs are absent from that checkpoint, so they start fresh: `lora_A` keeps
its kaiming initialization and `lora_B` stays zero. An adapted layer therefore
computes the pretrained projection exactly until training moves its adapters. See
[initialize policy weights](training.md#initialize-policy-weights) for the
supported layouts and the command.

## What the adapters cover

Without an explicit `target_modules`, every `nn.Linear` inside `double_blocks` and
`single_blocks` is adapted with PEFT LoRA: the pretrained projection becomes the
frozen `base_layer`, and a trainable rank-`r` `lora_A` plus a zero-initialized
`lora_B` pair plus dropout are added. The adapted set is exactly the attention and
feed-forward projections:

- Double blocks: `img_attn.qkv`, `img_attn.proj`, `img_mlp.0`, `img_mlp.2`,
  `txt_attn.qkv`, `txt_attn.proj`, `txt_mlp.0`, `txt_mlp.2`
- Single blocks: the fused `linear1` and `linear2`

That is 80 adapter targets and 160 adapter tensors for `klein-base-4b`
(5 double and 20 single blocks) and 112 targets for `klein-base-9b`
(8 double and 24 single blocks). Adaptation freezes the same number of projection
weights, 80 on the 4B variant, while the remaining 69 pretrained transformer
tensors stay trainable. The rank-64 default adds 92,405,760 adapter parameters on
the 4B variant and 165,675,008 on the 9B variant, which is `r * (in + out)` summed
over the adapted projections. The earlier layer inventory is recorded in
[dit-lora layer lists](agent-work/dit-lora/layer-lists.txt).

Loading order matters and is enforced: the official safetensors file is loaded
with `strict=True` and `assign=True` before any adapter is injected, so a renamed
or missing pretrained tensor still fails loudly. Initialization parity holds: a
zero-initialized LoRA up-projection leaves the residual exactly zero
(`module(probe) == module.get_base_layer()(probe)` bit for bit) and ones-initialized
IA3 vectors are exactly the identity, so a fresh adapter model matches the same
weights in full mode up to the float32 GEMM rounding that freezing the adapted
weights introduces; the measured forward difference is a few times 1e-07, while
any trained adapter moves the output by orders of magnitude more. The first
backward pass therefore leaves `lora_A` at zero gradient, which is expected.
Optimizers with decoupled weight decay still move that zero-gradient tensor, so a
strict "only the expected tensors moved" comparison uses `weight_decay=0`.

The transformer runs in training mode as usual. Because the official blocks
contain no dropout, batch normalization, or mode-dependent branch at the pinned
revision, training mode changes nothing for the frozen `base_layer` weights and the
trained blocks alike; only the injected adapter dropout reacts to it. Offline
evaluation calls `eval()` on the whole policy as before.

## Trainable and frozen parameters

`FluxAssetPolicy.parameter_report()` returns the exact named breakdown, and a
training run writes it to `parameter_report.json` in the output directory:

```python
report = policy.parameter_report()
report["mode"]                       # "dit_adapter" or "full"
report["dit_adapter"]                # backend, PEFT type and version, effective config, training policy
report["dit_lora"]                   # legacy LORA-shaped mirror, absent for IA3 and {"dit_lora": False} in full mode
report["adapters"]["layers"]         # adapted projection module paths
report["adapters"]["tensors"]        # lora_A/lora_B or ia3_l tensor names
report["fully_trainable"]["layers"]  # trained transformer layers and policy heads
report["frozen"]["groups"]           # per-scope counts and module paths
report["counts"]                     # adapter, trainable, frozen, and total tensors
```

Adapter tensors are classified with PEFT's own layer metadata, so an IA3 run lists
`ia3_l` parameters and never claims LoRA tensors. `dit_adapter` carries the
serialized effective PEFT configuration, so the report and the checkpoint contract
state exactly which adapter family, targets, and settings produced the run.

`frozen.groups` separates the adapted projection weights under
`policy.denoiser.transformer` from `autoencoder` and `text_encoder`, which makes a
mis-scoped parametrization obvious from one line of output. The same counts are
printed to the training log when the run starts. For `klein-base-4b` the adapter
count is fixed by the variant: 160 adapter tensors and 80 frozen projection
weights over the same 80 adapted projections, with the other 69 pretrained
transformer tensors listed under `fully_trainable` next to the policy heads.
Frozen totals grow by the autoencoder and text-encoder tensors supplied by the
assets.

## Checkpoints, resume, and EMA

Native `.safetensors` policy checkpoints keep the existing layout: a LoRA
checkpoint contains the loaded pretrained DiT weights plus the adapter tensors, so
`--initialize` and `load_policy_weights` roundtrip it exactly. Because the adapter
and base-layer names differ between modes, a full-mode checkpoint cannot be loaded
into an adapted model and an adapted checkpoint cannot be loaded into a full-mode
model; both cases fail with `Incomplete policy weights` or a shape error instead
of partially loading. A native IA3 checkpoint round-trips the same way.

Adapter settings are recorded in the model checkpoint contract as a `dit_adapter`
entry whenever an adapter is injected: the `peft` backend, the PEFT type and
version, the adapter target count, the layout, a serialized effective PEFT
configuration, and a `training_policy` marker
(`frozen_peft_adapted_projection_weights_remaining_dit_trainable`) that states
which parameters the optimizer is expected to see. Full-mode run signatures are
unchanged, so existing resume directories keep working. A resumed run with a
different adapter family, rank, alpha, dropout, targets, or PEFT version is
rejected by the usual continuation check, and a run recorded by the earlier custom
DiT LoRA implementation is rejected as well: its `dit_lora` record and
`base`/`residual_*` parameter names no longer match, which is intended because the
optimizer sees a different parameter set.

Checkpoints written by that earlier implementation can still initialize a new run.
`load_policy_weights` translates the legacy tensors onto the PEFT layout:
`<projection>.base.weight` becomes `<projection>.base_layer.weight`,
`residual_in.weight` becomes `lora_A.default.weight`, and `residual_out.weight`
becomes `lora_B.default.weight`. The translation runs only when the model carries
a PEFT LoRA adapter of the same rank at that projection, everything is validated
before any parameter is written, and legacy adapter tensors that cannot be placed
raise an error naming them instead of being dropped. Tensor-only checkpoints do
not record `lora_alpha`, so the alpha of the configured model stays in force; the
legacy and PEFT low-rank initializations are the same kaiming/zero pair, so a
matching rank and alpha reproduce the earlier function exactly.

EMA (`training.ema_decay`) keeps tracking trainable tensors only, so a LoRA run
averages the adapters, the trained transformer parameters, and the policy heads.
`ema_policy.safetensors` remains a complete policy state: the frozen projection
weights and encoder tensors are copied from the live weights and the trained
tensors come from the average.

## Validation

`tests/test_dit_lora.py` covers the contract on representative official FLUX.2
modules and needs no pretrained assets:

```bash
uv sync --extra dev --extra flux
PYTHONPATH="$FLUX2_SRC/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python -m pytest tests/test_dit_lora.py -q
```

The suite checks zero-init parity against an independently built full-mode model,
that adapted projection weights and encoders stay gradient-free while the
remaining transformer parameters receive gradients and move under a real
optimizer step, that the autoencoder and text encoder stay frozen and in
evaluation mode after `train()`, that the start-of-training hook changes no
parameter flag outside the adapted weights, the exact adapter/trained/frozen
report, invalid settings and every rejected option, default full-mode compatibility,
checkpoint roundtrip plus mode-mismatch rejection, legacy custom-LoRA translation,
an IA3 train/save/load/contract cycle, explicit `txt_in` targeting, dtype and device
preservation, the recorded 4B/9B inventories on meta tensors, the actionable
missing-dependency error, that full mode never imports PEFT, the `from_local_assets`
injection order, and an end-to-end training command that writes
`parameter_report.json`.

CPU checks establish the implementation contract only. Numerical quality,
memory, throughput, and benchmark scores of an adapter run against the full-finetune
recipes still require real-weight GPU verification. `klein-base-9b` shares the same
block layout and its adapter inventory is asserted on meta tensors, while the
forward-path tests exercise the 4B-shaped configuration.

## Optional variants

`dit_adapter_config` is the extension point. Narrowing the default inventory (for
example to attention projections alone, or to the single-stream blocks), adapting a
projection outside the blocks such as `txt_in`, or switching to IA3 changes the
adapter inventory, so the checkpoint contract changes with it and an incompatible
resume stays detectable. `FluxAssetPolicy.from_local_assets` and
`OfficialFluxDenoiser` both accept the mapping; the transformation lives in
`patchwam.models.lora`, which resolves and validates targets, dispatches the PEFT
configuration through `get_peft_config`, injects with `inject_adapter_in_model`,
and restores the original training flags.

The Qwen3-VL conditioning adapters remain on the local `LowRankResidualLinear`
primitive and are reported alongside the DiT adapters. They are out of scope for
`dit_adapter_config`.
