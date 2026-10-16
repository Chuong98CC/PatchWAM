# DiT LoRA: implementation report

Baseline `d2ebd3c6d9c46f9581b0ec46b27cdd39caf0bd80`. Status at the time of
writing: implementation, tests, and documentation complete; pending review.

## Changed paths

| Path | Change |
| --- | --- |
| `src/patchwam/models/lora.py` | New. `LowRankResidualLinear` plus block projection discovery/validation and injection |
| `src/patchwam/models/vision_language.py` | Imports the shared adapter primitive; no behavior change |
| `src/patchwam/models/flux.py` | DiT LoRA controls, freezing, encoder contract, `freeze_asset_encoders`, `parameter_report`, `from_local_assets` options |
| `src/patchwam/cli.py` | Writes `parameter_report.json` and logs the report counts when a run starts |
| `tests/test_dit_lora.py` | New. 24 CPU contract tests |
| `docs/dit_lora.md` | New. User documentation |
| `docs/index.md` | Guide-table row for the new document |

No preexisting user or worker changes were reverted. `configs/robotwin_vlm_5tasks.yaml`,
`docs/data.md`, `docs/training_recipes.md`, `src/patchwam/data/episodes.py`,
`src/patchwam/data/task_selection.py`, `tests/test_config_robotwin_5tasks.py`,
`tests/test_data_task_selection.py`, and `uv.lock` were left untouched.

## Contract as implemented (review round 2 amendment)

- `FluxAssetPolicy.from_local_assets(..., dit_lora=False, dit_lora_rank=64,
  dit_lora_alpha=128, dit_lora_dropout=0.05)`; the same four keywords on
  `OfficialFluxDenoiser`. Defaults leave the full-finetune path and its recorded
  model signature unchanged.
- LoRA mode wraps every `nn.Linear` below `double_blocks` and `single_blocks` in a
  zero-initialized low-rank residual. Wrapping freezes only each projection's
  `.base` weight, and the `residual_in`/`residual_out` adapters train. Every other
  pretrained transformer parameter keeps the original full-mode training
  contract: `img_in`, `txt_in`, `time_in`, both stream modulations,
  `single_stream_modulation`, `final_layer`, the attention QK-norm scales, and all
  remaining block weights. `policy.state_projection`,
  `policy.representation_projection`, and any other policy head outside the DiT
  stay fully trainable.
- The autoencoder and the entire text encoder are frozen and held in evaluation
  mode. `dit_lora=true` with `vl_trainable_adapters=true` raises
  `ValueError` (`src/patchwam/models/flux.py:493`).
- `rank`/`alpha`/`dropout` are validated with the argument named, in
  `patchwam.models.lora.validate_adapter_settings`.
- `checkpoint_contract()` adds `dit_lora` settings only when LoRA is enabled, so
  full-mode run signatures and existing resumes are unaffected while a changed
  LoRA rank/alpha/dropout/target count fails continuation. The record also carries
  `training_policy = DIT_LORA_TRAINING_POLICY`
  (`frozen_wrapped_block_bases_remaining_dit_trainable`), which separates this
  optimizer behavior from the earlier whole-transformer freeze.
- `OfficialFluxDenoiser.freeze_adapter_bases()` (renamed from
  `freeze_pretrained_weights`) re-freezes wrapped bases and re-asserts adapter
  training without touching any other parameter flag. `FluxAssetPolicy` calls it
  from the engine's `freeze_asset_encoders` hook, which also re-freezes the
  encoders and restores encoder evaluation mode.

## Verification

Environment: conda `fluxvla` (Python 3.10, torch 2.8.0+cu128, pytest 9.1.1) with
the pinned FLUX.2 checkout at revision `50fe516` on `PYTHONPATH`. All commands ran
from the repository root.

| Command | Result |
| --- | --- |
| `python -m pytest tests -q` on the baseline | 9 failed, 247 passed (`logs/baseline-suite.log`) |
| `python -m pytest tests/test_dit_lora.py -q` | 24 passed |
| `python -m pytest tests/test_dit_lora.py tests/test_models_flux.py tests/test_models.py tests/test_policy_weights.py tests/test_cli.py -q` | 69 passed |
| `python -m pytest tests -q` after the change | 9 failed, 271 passed (`logs/final-suite.log`); failure set identical to baseline |
| `python -m pytest tests/test_dit_lora.py tests/test_models_flux.py tests/test_models.py tests/test_conditioning_flow.py tests/test_policy_weights.py tests/test_averaging.py tests/test_engine.py tests/test_vision_history.py tests/test_training_validation.py -q` | 9 failed, 112 passed; same environment failures |
| `ruff check` on every changed file | All checks passed |
| Zero-init mutation check (`nn.init.zeros_` -> `nn.init.ones_` in `lora.py`) | 3 DiT LoRA tests plus the existing VL adapter test failed, then reverted |
| Review round 1: nonzero `residual_out` parity mutation (`nn.init.constant_(..., 0.25)`) against the corrected parity test | Failed at `tests/test_dit_lora.py:125`, the parity assertion (1536/1536 elements, max abs 0.12 against a 1e-6 tolerance), then reverted (`logs/parity-mutation.log`) |

### Review round 2 (contract amendment)

Commands use `OMP_NUM_THREADS=1` and the same environment as above.

| Command | Result |
| --- | --- |
| Round-2 baseline before editing: `pytest tests/test_dit_lora.py tests/test_models_flux.py tests/test_models.py tests/test_conditioning_flow.py tests/test_averaging.py tests/test_policy_weights.py tests/test_engine.py -q` | 6 failed (all `accelerate` resume mismatches), 106 passed (`logs/pre-change-contract.log`) |
| `pytest tests/test_dit_lora.py -q` after the amendment | 26 passed |
| Same focused set after the amendment | 6 failed, 108 passed (`logs/post-change-contract.log`); the same six environment failures |
| `ruff check` on `tests/test_dit_lora.py`, `models/flux.py`, `models/lora.py` | All checks passed |
| Parity mutation (`nn.init.constant_(residual_out.weight, 0.25)`) against the amended parity test | Failed at `tests/test_dit_lora.py:141`, the parity assertion (1536/1536 elements, max abs 0.090 against a 1e-6 tolerance), then reverted (`logs/parity-mutation.log`) |

The nine failures are environment drift, not regressions:

- `KeyError: 'step'` in `accelerate` 0.33.0 `load_accelerator_state` breaks every
  `--resume` test (6 in `test_averaging.py`, `test_conditioning_flow.py`,
  `test_engine.py`, `test_training_validation.py`). The project pins
  `accelerate==1.12.0`.
- `Qwen3VLForConditionalGeneration.get_image_features` returns three values in
  `transformers` 5.3.0 (`test_vision_history.py`, 2 tests). The project pins
  `transformers==4.57.1`.

## Correction after review round 1

The reviewer found that `test_adapter_is_zero_initialized_and_the_frozen_backbone_is_unchanged`
built the adapted denoiser on `plain.transformer` itself, so in-place injection
mutated the shared core and both the expected and actual forwards ran through the
adapters. The parity assertion was vacuous.

Fix: the adapted model now wraps `copy.deepcopy(plain.transformer)`, the test
asserts that `plain.transformer` contains no `LowRankResidualLinear` after
injection, and the expected output comes from the genuinely unadapted `plain`
model (`tests/test_dit_lora.py:113`).

Investigating the corrected comparison produced one implementation fact worth
recording. The injected residual is exactly zero, but freezing a weight changes
PyTorch's forward GEMM path, so a full-mode and a LoRA-mode forward of identical
weights differ by roughly 2e-07 to 4e-07 in float32 across runs. The isolated
adapter stays bitwise inert. Re-measured after the round-2 amendment, which
freezes only the wrapped bases (`logs/parity-diagnosis.log`):

```text
wrapped-base freeze effect: False 3.5762786865234375e-07
adapters bitwise inert: True 20
frozen tensors: 20 of 81
frozen non-base tensors: []
```

The test therefore asserts end-to-end parity with `rtol=1e-5, atol=1e-6`, which
is roughly an order of magnitude above the freeze-induced rounding and far below
any trained adapter, plus an exact-zero assertion on every adapter's residual
branch and a bitwise `module(probe) == module.base(probe)` check. The targeted
mutation check confirms the tolerance still has teeth: initializing
`residual_out` to 0.25 failed the parity assertion with 1536/1536 differing
elements and a maximum absolute difference of 0.12 at that revision. The amended
contract was re-checked the same way, see the round-2 table below.

Re-verified after the fix: `pytest tests/test_dit_lora.py -q` reports 24 passed
with the mutation reverted. No other test file was rerun, and no unrelated change
was made.

## Exact layer inventory

`layer-lists.txt` records the generated lists for the official meta-device
configurations:

| Variant | Adapter targets | Adapter tensors (trained) | Frozen wrapped bases | Trained pretrained DiT tensors |
| --- | --- | --- | --- | --- |
| `klein-base-4b` | 80 | 160 | 80 | 69 |
| `klein-base-9b` | 112 | 224 | 112 | 89 |

Wrapped projection paths (4B, five double and twenty single blocks):

```text
double_blocks.{0..4}.img_attn.qkv
double_blocks.{0..4}.img_attn.proj
double_blocks.{0..4}.img_mlp.0
double_blocks.{0..4}.img_mlp.2
double_blocks.{0..4}.txt_attn.qkv
double_blocks.{0..4}.txt_attn.proj
double_blocks.{0..4}.txt_mlp.0
double_blocks.{0..4}.txt_mlp.2
single_blocks.{0..19}.linear1
single_blocks.{0..19}.linear2
```

Frozen in LoRA mode: the 80 wrapped projection weights (`*.base.weight`) for 4B,
plus every autoencoder and text-encoder tensor.

Trained pretrained DiT layers, unchanged from full mode (69 tensors for 4B, 89 for
9B): `img_in`, `time_in.in_layer`, `time_in.out_layer`, `txt_in`,
`double_stream_modulation_img.lin`, `double_stream_modulation_txt.lin`,
`single_stream_modulation.lin`, `final_layer.linear`,
`final_layer.adaLN_modulation.1`, and inside every block the QK-norm scales
`{img,txt}_attn.norm.{query,key}_norm.scale` (double) and
`norm.{query,key}_norm.scale` (single). `pe_embedder` and the block layer norms
carry no parameters at this revision.

Trained policy heads, outside the DiT:

```text
policy.state_projection                (weight, bias)                always when proprio_dim is set
policy.representation_projection.0     (weight, bias)                Self-Flow variants only
policy.representation_projection.2     (weight, bias)                Self-Flow variants only
```

Adapter tensors per wrapped projection: `<projection>.residual_in.weight` and
`<projection>.residual_out.weight`.

## Optional variants (not implemented)

Only the default target set ships. Narrowing or extending it (attention
projections only, single-stream blocks only, or adding modulation/final-layer
adapters) would change the adapter inventory and therefore the checkpoint
contract. The extension point is `patchwam.models.lora.attach_block_adapters`,
which currently selects every `nn.Linear` below `double_blocks` and
`single_blocks`; a variant must keep the settings validation, the
zero-initialized output projection, and an accurate `adapter_targets` count so
resume mismatches remain detectable.

## Caveats and residual risk

- Every check is CPU-only with synthetic official micro-modules. Numerical
  quality, throughput, memory, and benchmark parity against the full-finetune
  recipes are unverified.
- `text_encoder_kind="qwen3_vl"` in LoRA mode is covered by the wrapper-level
  freeze/evaluation test and by construction-time rejection of trainable VL
  adapters, but not by an end-to-end VL encode inside this environment, because
  the pinned Qwen3-VL adapter tests already fail on `transformers` 5.3.0.
- A resumed run that switches between full and LoRA mode fails as intended, but
  the engine resume path itself cannot be exercised here (`accelerate` 0.33.0).
- The default rank/alpha/dropout follow the existing VL adapter convention
  (64/128/0.05); no tuning evidence supports those values for the DiT.
- The transformer stays in training mode during optimization, as in full mode. At
  the pinned revision `flux2/model.py` contains no dropout, batch normalization,
  or `self.training` branch, so training mode is inert for the frozen projection
  bases and the trained blocks alike and only the adapter dropout responds to it;
  the autoencoder and text encoders are forced to evaluation mode explicitly.
