# DiT LoRA finetuning: design and phase plan

Task: optional LoRA finetuning of the official FLUX DiT backbone. Vision and
language encoders stay frozen in this mode; every other learnable policy layer is
fully trained. Baseline `d2ebd3c6d9c46f9581b0ec46b27cdd39caf0bd80`.

## Contract

- `FluxAssetPolicy.from_local_assets` gains `dit_lora`, `dit_lora_rank`,
  `dit_lora_alpha`, `dit_lora_dropout`. Defaults (`False`) reproduce the current
  full-finetuning behavior with an unchanged model signature.
- `OfficialFluxDenoiser` gains the same controls and performs injection, because it
  owns the official layer contract.
- Pretrained weights load with `strict=True` before injection, so a wrapper never
  hides a missing or renamed official tensor.
- LoRA mode wraps every `nn.Linear` inside `double_blocks` and `single_blocks`
  with a zero-initialized low-rank residual (`LowRankResidualLinear`, reused from
  the VL adapter path and moved to `models/lora.py`). Wrapping freezes only each
  projection's base weight. `img_in`, `txt_in`, `time_in`, `pe_embedder`, both
  stream modulations, `final_layer`, the QK-norm scales, and every other
  pretrained DiT parameter stay fully trainable, exactly as in full mode.
- `policy.state_projection`, an optional `policy.representation_projection`, and
  any other learnable policy head outside the DiT remain fully trainable.
- The autoencoder and the complete text encoder (including VL adapters) freeze and
  stay in evaluation mode. `dit_lora=True` with `vl_trainable_adapters=True` is
  rejected instead of silently ignoring one of the two requests.
- LoRA settings are recorded in `FluxAssetPolicy.checkpoint_contract()`, and only
  when LoRA is enabled, so full-mode run signatures are unchanged while a
  rank/alpha/dropout change still blocks `--resume`. The record carries a
  `training_policy` marker (`DIT_LORA_TRAINING_POLICY`) so a run recorded under
  the earlier whole-transformer freeze cannot resume under the new optimizer
  behavior.
- `FluxAssetPolicy.parameter_report()` names adapter tensors, fully trained layers,
  and frozen layers. The training CLI writes it to `parameter_report.json`.

### Contract amendment (user-authorized, review round 2)

The first implementation froze the entire pretrained DiT. The accepted contract is
now "original training everywhere except the wrapped block Linear weights": only
`<projection>.base.weight` is frozen, its `residual_in`/`residual_out` train, and
every other DiT parameter keeps its original trainable state. The start-of-training
hook changed accordingly: it re-freezes wrapped bases and re-asserts adapter
training without touching any other parameter flag.

## Phases

1. Discovery: repository guidance, model/engine/checkpoint contracts, official
   `flux2.model` block layout, baseline test state. Done.
2. Extract the adapter primitive to `models/lora.py`; add block projection
   discovery plus validated injection.
3. Wire `OfficialFluxDenoiser` and `from_local_assets`; freeze encoders; add the
   checkpoint contract and parameter report.
4. CPU tests in `tests/test_dit_lora.py` with representative official dummy FLUX
   modules: zero-init parity, gradient/optimizer isolation, encoder freezing,
   invalid configuration, default compatibility, checkpoint roundtrip, and the
   local-asset factory path.
5. Documentation in `docs/dit_lora.md`; run the existing relevant suites; record
   command output under `logs/`.

## Verification plan

`tests/test_dit_lora.py` plus `tests/test_models_flux.py`,
`tests/test_models.py`, `tests/test_conditioning_flow.py`,
`tests/test_policy_weights.py`, `tests/test_averaging.py`, `tests/test_engine.py`,
and `tests/test_vision_history.py`. The nine baseline failures are environment
drift (`accelerate` 0.33.0 vs the pinned 1.12.0 resume API, `transformers` 5.3.0
vs the pinned 4.57.1 VL `get_image_features` signature) and are unrelated to this
feature; see `logs/baseline-suite.log`.

## Checkpoint

All five phases are complete. Implementation, the 24 CPU contract tests,
documentation, and the verification runs are finished; results are recorded in
`report.md`, `layer-lists.txt`, and `logs/`. Pending Astra review.
