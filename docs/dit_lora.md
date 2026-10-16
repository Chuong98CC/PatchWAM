# Optional DiT LoRA finetuning

[Overview](../README.md) · [Model assets](models.md) · [Training](training.md) · [Conditioning and training options](conditioning_and_training.md)

The default recipe fully finetunes the FLUX.2 denoising transformer. The optional
DiT LoRA mode keeps that recipe and replaces the block attention and feed-forward
projections with frozen-weight, low-rank residual adapters. Only the wrapped
projection weights are frozen; every other pretrained transformer parameter keeps
training exactly as it does without adapters.

## Freeze contract

| Component | LoRA mode | Full mode (default) |
| --- | --- | --- |
| Wrapped block projection weights (`double_blocks`, `single_blocks` `.base`) | Frozen | Fully trained |
| Block projection adapters (`residual_in`, `residual_out`) | Trained | Not present |
| `img_in`, `txt_in`, `time_in`, stream modulations, `final_layer`, attention QK-norm scales, every other block weight | Fully trained | Fully trained |
| `policy.state_projection` | Fully trained | Fully trained |
| `policy.representation_projection` (Self-Flow variants) | Fully trained | Fully trained |
| Autoencoder | Frozen, evaluation mode | Frozen, evaluation mode |
| Text encoder, including VL adapters | Frozen, evaluation mode | Frozen, or trained when `vl_trainable_adapters=true` |

`dit_lora=true` with `vl_trainable_adapters=true` fails at construction instead of
silently ignoring one of the two requests:

```text
ValueError: dit_lora freezes the complete text encoder, including VL adapters;
set vl_trainable_adapters=false or disable dit_lora
```

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
| `model.dit_lora` | `false` | Freeze the wrapped block projection weights and train their adapters alongside the rest of the DiT |
| `model.dit_lora_rank` | `64` | Adapter bottleneck width; a positive integer |
| `model.dit_lora_alpha` | `128` | Residual multiplier `alpha / rank`; finite and positive |
| `model.dit_lora_dropout` | `0.05` | Dropout on the adapter input path, in `[0,1)` |

`OfficialFluxDenoiser` accepts the same four keyword arguments directly, which is
the API the CPU layer-contract tests use. Invalid settings raise
`ValueError` naming the offending argument, for example
`dit_lora_rank must be a positive integer`.

## What the adapters cover

Every `nn.Linear` inside `double_blocks` and `single_blocks` is wrapped with a
frozen base, a trainable `residual_in` projection, a trainable zero-initialized
`residual_out` projection, and dropout. The wrapped set is exactly the attention
and feed-forward projections:

- Double blocks: `img_attn.qkv`, `img_attn.proj`, `img_mlp.0`, `img_mlp.2`,
  `txt_attn.qkv`, `txt_attn.proj`, `txt_mlp.0`, `txt_mlp.2`
- Single blocks: the fused `linear1` and `linear2`

That is 80 adapter targets and 160 adapter tensors for `klein-base-4b`
(5 double and 20 single blocks) and 112 targets for `klein-base-9b`
(8 double and 24 single blocks). Wrapping freezes the same number of projection
weights, 80 on the 4B variant, while the remaining 69 pretrained transformer
tensors stay trainable. The full three-way inventory is recorded in
[dit-lora layer lists](agent-work/dit-lora/layer-lists.txt).

Loading order matters and is enforced: the official safetensors file is loaded
with `strict=True` and `assign=True` before any adapter is injected, so a renamed
or missing pretrained tensor still fails loudly. Initialization parity holds: the
injected residual is exactly zero (`module(probe) == module.base(probe)` bit for
bit), so a fresh LoRA model matches the same weights in full mode up to the
float32 GEMM rounding that freezing the base weights introduces; the measured
forward difference is a few times 1e-07, while any trained adapter moves the
output by orders of magnitude more. The first backward pass therefore leaves `residual_in`
at zero gradient, which is expected.
Optimizers with decoupled weight decay still move that zero-gradient tensor, so a
strict "only the expected tensors moved" comparison uses `weight_decay=0`.

The transformer runs in training mode as usual. Because the official blocks
contain no dropout, batch normalization, or mode-dependent branch at the pinned
revision, training mode changes nothing for the frozen projection bases and the
trained blocks alike; only the injected adapter dropout reacts to it. Offline
evaluation calls `eval()` on the whole policy as before.

## Trainable and frozen parameters

`FluxAssetPolicy.parameter_report()` returns the exact named breakdown, and a
training run writes it to `parameter_report.json` in the output directory:

```python
report = policy.parameter_report()
report["mode"]                       # "dit_lora" or "full"
report["dit_lora"]                   # rank, alpha, dropout, target count, training policy
report["adapters"]["layers"]         # wrapped projection module paths
report["adapters"]["tensors"]        # residual_in and residual_out tensor names
report["fully_trainable"]["layers"]  # trained transformer layers and policy heads
report["frozen"]["groups"]           # per-scope counts and module paths
report["counts"]                     # adapter, trainable, frozen, and total tensors
```

`frozen.groups` separates the wrapped projection bases under
`policy.denoiser.transformer` from `autoencoder` and `text_encoder`, which makes a
mis-scoped parametrization obvious from one line of output. The same counts are
printed to the training log when the run starts. For `klein-base-4b` the adapter
count is fixed by the variant: 160 adapter tensors and 80 frozen projection
weights over the same 80 wrapped projections, with the other 69 pretrained
transformer tensors listed under `fully_trainable` next to the policy heads.
Frozen totals grow by the autoencoder and text-encoder tensors supplied by the
assets.

## Checkpoints, resume, and EMA

Native `.safetensors` policy checkpoints keep the existing layout: a LoRA
checkpoint contains the loaded pretrained DiT weights plus the adapter tensors, so
`--initialize` and `load_policy_weights` roundtrip it exactly. Because the adapter
and wrapped-base names differ between modes, a full-mode checkpoint cannot be
loaded into a LoRA model and a LoRA checkpoint cannot be loaded into a full-mode
model; both cases fail with `Incomplete policy weights` or a shape error instead
of partially loading.

LoRA settings are recorded in the model checkpoint contract when LoRA is enabled:
rank, alpha, dropout, the adapter target count, and a `training_policy` marker
(`frozen_wrapped_block_bases_remaining_dit_trainable`) that states which
parameters the optimizer is expected to see. Full-mode run signatures are
unchanged, so existing resume directories keep working. A resumed LoRA run with
different rank, alpha, dropout, or adapter targets is rejected by the usual
continuation check, and a run recorded under an earlier policy that froze the
whole transformer is rejected too, because the marker changes the model
signature.

EMA (`training.ema_decay`) keeps tracking trainable tensors only, so a LoRA run
averages the adapters, the trained transformer parameters, and the policy heads.
`ema_policy.safetensors` remains a complete policy state: the frozen projection
weights and encoder tensors are copied from the live weights and the trained
tensors come from the average.

## Validation

`tests/test_dit_lora.py` covers the contract on representative official FLUX.2
modules and needs no pretrained assets:

```bash
PYTHONPATH="$FLUX2_SRC/src${PYTHONPATH:+:$PYTHONPATH}" python -m pytest tests/test_dit_lora.py -q
```

The suite checks zero-init parity against an independently built full-mode model,
that wrapped projection weights and encoders stay gradient-free while the
remaining transformer parameters receive gradients and move under a real
optimizer step, that the autoencoder and text encoder stay frozen and in
evaluation mode after `train()`, that the start-of-training hook changes no
parameter flag outside the wrapped bases, the exact adapter/trained/frozen
report, invalid settings, default full-mode compatibility, checkpoint roundtrip
plus mode-mismatch rejection, the `from_local_assets` injection order, and an
end-to-end training command that writes `parameter_report.json`.

CPU checks establish the implementation contract only. Numerical quality,
memory, throughput, and benchmark scores of a LoRA run against the full-finetune
recipes still require real-weight GPU verification. `klein-base-9b` shares the
same block layout and is covered by the shared layer-contract code, but the tests
exercise the 4B-shaped configuration.

## Optional variants

Only the default target set above is implemented. Narrowing it (for example to
attention projections alone, or to the single-stream blocks) would change the
adapter inventory and therefore the checkpoint contract. The extension point is
`patchwam.models.lora.attach_block_adapters`, which currently wraps every
`nn.Linear` below `double_blocks` and `single_blocks`; a variant must keep the
`validate_adapter_settings` checks, the zero-initialized output projection, and
the recorded `adapter_targets` count and `training_policy` marker so resume
mismatches stay detectable.
