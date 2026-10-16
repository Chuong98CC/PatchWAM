"""Optional DiT LoRA finetuning on representative official FLUX.2 modules."""

import copy
import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from patchwam.checkpoints import load_policy_weights, policy_weight_state
from patchwam.models import (
    FluxAssetPolicy,
    OfficialFluxDenoiser,
    joint_visibility,
    sequence_coordinates,
)
from patchwam.models.flux import DIT_LORA_TRAINING_POLICY
from patchwam.models.lora import LowRankResidualLinear
from patchwam.models.policy import PatchFlowPolicy, make_tiny_policy

api = pytest.importorskip("flux2.model", reason="Install the official FLUX.2 dependency for its adapter tests")

LORA = {"dit_lora": True, "dit_lora_rank": 2, "dit_lora_alpha": 4, "dit_lora_dropout": 0}
DIT_PREFIX = "policy.denoiser.transformer."

# Representative DiT parameters the contract keeps training outside the adapters.
TRAINED_DIT = (
    "img_in.weight", "txt_in.weight", "time_in.in_layer.weight", "time_in.out_layer.weight",
    "double_stream_modulation_img.lin.weight", "double_stream_modulation_txt.lin.weight",
    "single_stream_modulation.lin.weight", "final_layer.linear.weight",
    "final_layer.adaLN_modulation.1.weight",
    "double_blocks.0.img_attn.norm.query_norm.scale", "single_blocks.0.norm.key_norm.scale",
)
# Representative wrapped projection bases the contract freezes.
FROZEN_DIT = (
    "double_blocks.0.img_attn.qkv.base.weight", "double_blocks.0.txt_mlp.0.base.weight",
    "single_blocks.0.linear1.base.weight", "single_blocks.1.linear2.base.weight",
)


def official_core():
    config = api.Klein4BParams(
        context_in_dim=32, hidden_size=64, num_heads=4, depth=2, depth_single_blocks=2,
        axes_dim=[4, 4, 4, 4], mlp_ratio=2,
    )
    return api.Flux2(config)


def denoiser_inputs(reference_length=3):
    return {
        "reference": torch.randn(2, reference_length, 128),
        "noisy": torch.randn(2, 6, 128), "context": torch.randn(2, 4, 32),
        "sigma": torch.tensor([0.3, 0.7]),
        "reference_ids": sequence_coordinates(2, reference_length, group=10),
        "noisy_ids": sequence_coordinates(2, 6, group=20),
        "context_ids": sequence_coordinates(2, 4),
        "visibility": joint_visibility(4, reference_length, 4, 2),
    }


def base_weight_name(name):
    """Map a wrapped official parameter name back to its pretrained name."""
    return name.replace(".base.weight", ".weight").replace(".base.bias", ".bias")


class _Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 128, 1)

    def encode(self, images):
        return F.adaptive_avg_pool2d(self.projection(images), (2, 2))


class _TextEncoder(nn.Module):
    requires_images = True
    trainable_adapters = True

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(3, 32))
        self.modes = []

    def forward(self, prompts, *, current, history=None, history_valid=None, subtasks=None):
        count = len(prompts)
        return {
            "text_tokens": current.mean((1, 2, 3))[:, None, None].expand(count, 2, 32) * self.embeddings[0, 0],
            "text_valid": torch.ones(count, 2, dtype=torch.bool),
        }

    def train(self, mode: bool = True):
        self.modes.append(bool(mode))
        return super().train(mode)


class _PlainTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(3, 32))

    def forward(self, prompts):
        return self.embeddings.unsqueeze(0).expand(len(prompts), -1, -1)


def video_sample():
    return {
        "video": torch.randn(2, 3, 2, 16, 16), "prompt": ["move", "lift"],
        "action": torch.randn(2, 16, 14), "proprio": torch.randn(2, 14),
    }


def lora_policy(**changes):
    options = LORA | changes
    denoiser = OfficialFluxDenoiser(official_core(), **options)
    return PatchFlowPolicy(denoiser, action_dim=14, text_dim=32, proprio_dim=14)


def lora_wrapper(text_encoder=None, **changes):
    return FluxAssetPolicy(
        lora_policy(**changes), _Autoencoder(),
        _PlainTextEncoder() if text_encoder is None else text_encoder,
    )


def test_adapter_injection_is_zero_initialized_and_leaves_the_rest_trainable():
    plain = OfficialFluxDenoiser(official_core())
    untouched = copy.deepcopy(plain.transformer)
    adapted = OfficialFluxDenoiser(copy.deepcopy(plain.transformer), **LORA)
    # Injection must leave the full-mode model alone; parity is only meaningful
    # against an independently wrapped, genuinely unadapted backbone.
    assert not any(isinstance(module, LowRankResidualLinear) for module in plain.transformer.modules())
    data = denoiser_inputs()
    expected = plain(**dict(data)).detach()
    # The injected residual is exactly zero, so the outputs differ only by the
    # float32 GEMM rounding that freezing the wrapped weights introduces (~1e-7
    # here). The tolerance is tight enough that any trained adapter fails it.
    torch.testing.assert_close(adapted(**dict(data)), expected, rtol=1e-5, atol=1e-6)
    probes = 0
    for module in adapted.transformer.modules():
        if isinstance(module, LowRankResidualLinear):
            probe = torch.randn(3, module.base.in_features)
            residual = module.residual_out(module.residual_in(module.dropout(probe))) * module.multiplier
            assert torch.equal(residual, torch.zeros_like(residual))
            assert torch.equal(module(probe), module.base(probe))
            probes += 1
    assert probes == len(adapted.lora_targets)
    transformed = {
        base_weight_name(name): parameter for name, parameter in adapted.transformer.named_parameters()
    }
    pretrained = dict(untouched.named_parameters())
    for name, parameter in pretrained.items():
        torch.testing.assert_close(transformed[base_weight_name(name)], parameter, rtol=0, atol=0)
    assert len(transformed) == len(pretrained) + 2 * len(adapted.lora_targets)
    # Only the wrapped bases are frozen; the remaining pretrained DiT parameters
    # keep the full-mode training contract.
    adapted(**dict(data)).square().mean().backward()
    for name, parameter in adapted.transformer.named_parameters():
        if "residual" in name:
            assert parameter.requires_grad and parameter.grad is not None
        elif ".base." in name:
            assert not parameter.requires_grad and parameter.grad is None, name
        else:
            assert parameter.requires_grad and parameter.grad is not None, name
    # The zero-initialized output projection leaves the first-step input gradient at zero.
    assert all(
        parameter.grad.abs().sum() == 0
        for name, parameter in adapted.transformer.named_parameters()
        if name.endswith("residual_in.weight")
    )


def test_full_mode_defaults_keep_the_original_module_layout_trainable():
    core = official_core()
    keys = set(core.state_dict())
    denoiser = OfficialFluxDenoiser(core)
    assert denoiser.dit_lora is False and denoiser.dit_lora_contract() is None
    assert set(core.state_dict()) == keys
    assert not any(isinstance(module, LowRankResidualLinear) for module in denoiser.modules())
    assert all(parameter.requires_grad for parameter in denoiser.parameters())
    assert torch.isfinite(denoiser(**denoiser_inputs())).all()


def test_construction_and_the_hook_change_only_wrapped_base_flags():
    core = official_core()
    reference = {name: parameter.requires_grad for name, parameter in core.named_parameters()}
    assert reference and all(reference.values())
    assert not any("residual" in name for name in reference)
    denoiser = OfficialFluxDenoiser(core, **LORA)
    targets = set(denoiser.lora_targets)
    flags = {name: parameter.requires_grad for name, parameter in core.named_parameters()}
    adapters = [name for name in flags if "residual_" in name]
    bases = [name for name in flags if name.endswith(".base.weight")]
    untouched = {
        name: flag for name, flag in flags.items() if name not in set(adapters) and name not in set(bases)
    }
    assert sorted(bases) == sorted(f"{target}.base.weight" for target in targets)
    assert len(adapters) == 2 * len(targets)
    assert all(flags[name] for name in adapters)
    assert not any(flags[name] for name in bases)
    # Injection leaves every parameter it did not wrap exactly as it was.
    assert untouched == {
        name: flag for name, flag in reference.items() if name.rsplit(".", 1)[0] not in targets
    }
    denoiser.freeze_adapter_bases()
    assert {name: parameter.requires_grad for name, parameter in core.named_parameters()} == flags
    with pytest.raises(ValueError, match="not configured for DiT LoRA"):
        OfficialFluxDenoiser(official_core()).freeze_adapter_bases()


def test_block_adapters_cover_every_attention_and_mlp_projection_once():
    denoiser = OfficialFluxDenoiser(official_core(), **LORA)
    targets = denoiser.lora_targets
    assert [name.split(".")[0] for name in targets] == ["double_blocks"] * 16 + ["single_blocks"] * 4
    assert len(set(targets)) == len(targets) == 20
    assert sum(name.startswith("double_blocks") for name in targets) == 2 * 8
    assert sum(name.startswith("single_blocks") for name in targets) == 2 * 2
    leaves = {name.rsplit(".", 1)[-1] for name in targets}
    assert leaves == {"qkv", "proj", "linear1", "linear2", "0", "2"}
    for name in targets:
        assert isinstance(denoiser.transformer.get_submodule(name), LowRankResidualLinear)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"dit_lora_rank": 0}, "dit_lora_rank"),
        ({"dit_lora_rank": -3}, "dit_lora_rank"),
        ({"dit_lora_rank": 2.5}, "dit_lora_rank"),
        ({"dit_lora_alpha": 0}, "dit_lora_alpha"),
        ({"dit_lora_alpha": float("inf")}, "dit_lora_alpha"),
        ({"dit_lora_dropout": -0.1}, "dit_lora_dropout"),
        ({"dit_lora_dropout": 1}, "dit_lora_dropout"),
        ({"dit_lora": None}, "dit_lora must be a boolean"),
    ],
)
def test_invalid_lora_settings_are_rejected_with_the_offending_argument(changes, message):
    with pytest.raises(ValueError, match=message):
        OfficialFluxDenoiser(official_core(), **(LORA | changes))


def test_only_wrapped_bases_and_encoders_stay_gradient_free():
    wrapper = lora_wrapper()
    wrapper.train()
    wrapper(video_sample())["loss"].backward()
    parameters = dict(wrapper.named_parameters())
    for name, parameter in wrapper.named_parameters():
        if name.startswith(("autoencoder", "text_encoder")) or ".base." in name:
            assert not parameter.requires_grad and parameter.grad is None, name
        else:
            assert parameter.requires_grad, name
    for name in TRAINED_DIT:
        gradient = parameters[DIT_PREFIX + name].grad
        assert gradient is not None and gradient.abs().sum() > 0, name
    for name in FROZEN_DIT:
        assert parameters[DIT_PREFIX + name].grad is None, name
    assert parameters["policy.state_projection.weight"].grad.abs().sum() > 0
    assert all(parameters["policy." + name].grad is not None for name in (
        "denoiser.transformer.double_blocks.0.img_attn.qkv.residual_out.weight",
        "denoiser.transformer.single_blocks.0.linear2.residual_out.weight",
    ))


def test_optimizer_step_moves_adapters_and_every_unwrapped_dit_parameter():
    torch.manual_seed(5)
    wrapper = lora_wrapper()
    wrapper.train()
    before = {name: tensor.detach().clone() for name, tensor in wrapper.named_parameters()}
    optimizer = torch.optim.AdamW(
        [parameter for parameter in wrapper.parameters() if parameter.requires_grad],
        lr=1e-2, weight_decay=0,
    )
    optimizer.zero_grad(set_to_none=True)
    wrapper(video_sample())["loss"].backward()
    optimizer.step()
    changed = {name for name, tensor in wrapper.named_parameters() if not torch.equal(tensor, before[name])}
    assert all(
        name in changed for name in
        ["policy.state_projection.weight"] + [DIT_PREFIX + name for name in TRAINED_DIT]
    ), sorted(set(changed) - {DIT_PREFIX + name for name in TRAINED_DIT})
    assert sum("residual_out" in name for name in changed) == 20
    # AdamW does not move the zero-gradient first-step input projection.
    assert not any("residual_in" in name for name in changed)
    assert not any(".base." in name for name in changed)
    assert not any(name.startswith(("autoencoder", "text_encoder")) for name in changed)


def test_encoders_stay_frozen_and_the_training_hook_keeps_only_bases_frozen():
    wrapper = lora_wrapper(text_encoder=_TextEncoder(), dit_lora_dropout=0.1)
    wrapper.train()
    assert wrapper.policy.training and wrapper.policy.denoiser.training
    assert not wrapper.autoencoder.training and not wrapper.text_encoder.training
    assert wrapper.text_encoder.modes[-1] is False
    assert wrapper.policy.state_projection.training
    adapters = [module for module in wrapper.policy.denoiser.modules() if isinstance(module, nn.Dropout)]
    assert adapters and all(module.training for module in adapters)
    flags = {name: parameter.requires_grad for name, parameter in wrapper.named_parameters()}
    wrapper.freeze_asset_encoders()
    # The start-of-training hook re-freezes wrapped bases and encoders only.
    assert {name: parameter.requires_grad for name, parameter in wrapper.named_parameters()} == flags
    assert not wrapper.text_encoder.training and wrapper.text_encoder.modes[-1] is False
    assert next(module for module in adapters).training
    assert all(
        not parameter.requires_grad
        for name, parameter in wrapper.named_parameters()
        if name.startswith(("autoencoder", "text_encoder"))
    )
    parameters = dict(wrapper.named_parameters())
    assert all(parameters[DIT_PREFIX + name].requires_grad for name in TRAINED_DIT)
    assert all(not parameters[DIT_PREFIX + name].requires_grad for name in FROZEN_DIT)


def test_full_mode_still_trains_image_conditioned_encoders():
    encoder = _TextEncoder()
    wrapper = FluxAssetPolicy(make_tiny_policy(), _Autoencoder(), encoder)
    assert wrapper.dit_lora is False
    wrapper.train()
    assert encoder.modes[-1] is True and wrapper.text_encoder.training


def test_self_flow_representation_projection_outside_the_dit_stays_trainable():
    denoiser = OfficialFluxDenoiser(official_core(), **LORA)
    wrapper = FluxAssetPolicy(
        PatchFlowPolicy(denoiser, action_dim=14, text_dim=32, proprio_dim=14, self_flow_variant=2),
        _Autoencoder(), _PlainTextEncoder(),
    )
    report = wrapper.parameter_report()
    trainable = report["fully_trainable"]["layers"]
    for layer in ("policy.representation_projection.0", "policy.representation_projection.2",
                  "policy.state_projection", DIT_PREFIX + "img_in"):
        assert layer in trainable
    # 21 unwrapped DiT tensors, 2 state-projection tensors, 4 representation tensors.
    assert report["counts"]["fully_trainable"] == 27
    wrapper.train()
    assert wrapper.policy.representation_projection.training
    wrapper.freeze_asset_encoders()
    assert all(
        parameter.requires_grad
        for name, parameter in wrapper.named_parameters()
        if name.startswith("policy.representation_projection")
    )


def test_parameter_report_separates_adapters_full_and_frozen_layers():
    wrapper = lora_wrapper()
    report = wrapper.parameter_report()
    assert json.loads(json.dumps(report)) == report
    assert report["mode"] == "dit_lora"
    assert report["dit_lora"] == {
        "dit_lora": True, "rank": 2, "alpha": 4, "dropout": 0,
        "adapter_targets": len(wrapper.policy.denoiser.lora_targets),
        "adapter_layout": "block_attention_and_mlp_projections",
        "training_policy": DIT_LORA_TRAINING_POLICY,
    }
    layers = report["adapters"]["layers"]
    assert "policy.denoiser.transformer.double_blocks.1.txt_mlp.2" in layers
    assert "policy.denoiser.transformer.single_blocks.0.linear1" in layers
    assert report["counts"]["adapters"] == 2 * len(layers) == 40
    trained = set(report["fully_trainable"]["tensors"])
    for name in TRAINED_DIT:
        assert DIT_PREFIX + name in trained, name
    assert {"img_in", "txt_in", "final_layer.linear", "double_stream_modulation_img.lin"} <= {
        entry.removeprefix(DIT_PREFIX) for entry in report["fully_trainable"]["layers"]
    }
    assert {"policy.state_projection", "policy.denoiser.transformer.img_in"} <= set(
        report["fully_trainable"]["layers"]
    )
    frozen = set(report["frozen"]["tensors"])
    for name in FROZEN_DIT:
        assert DIT_PREFIX + name in frozen, name
    assert all(name.endswith(".base.weight") for name in frozen if name.startswith(DIT_PREFIX))
    assert not any(name.startswith(DIT_PREFIX) and ".base." not in name for name in frozen)
    groups = report["frozen"]["groups"]
    assert {scope: group["tensor_count"] for scope, group in groups.items()} == {
        "autoencoder": 2, "policy.denoiser.transformer": 20, "text_encoder": 1,
    }
    assert len(groups["policy.denoiser.transformer"]["layers"]) == 20
    assert all(layer.endswith(".base") for layer in groups["policy.denoiser.transformer"]["layers"])
    assert sum(group["parameter_count"] for group in groups.values()) == report["frozen"]["parameter_count"]
    assert report["counts"] == {
        "adapters": 40, "fully_trainable": 23, "trainable": 63, "frozen": 23, "total": 86,
    }
    assert report["counts"]["total"] == report["counts"]["trainable"] + report["counts"]["frozen"]


def test_full_mode_parameter_report_keeps_the_dit_fully_trainable():
    wrapper = FluxAssetPolicy(PatchFlowPolicy(
        OfficialFluxDenoiser(official_core()), action_dim=14, text_dim=32, proprio_dim=14,
    ), _Autoencoder(), _PlainTextEncoder())
    report = wrapper.parameter_report()
    assert report["mode"] == "full" and report["dit_lora"] == {"dit_lora": False}
    assert report["adapters"]["tensor_count"] == 0
    assert "policy.denoiser.transformer.img_in" in report["fully_trainable"]["layers"]
    assert report["frozen"]["groups"].keys() == {"autoencoder", "text_encoder"}


def test_checkpoint_contract_records_lora_settings_and_leaves_full_mode_unchanged():
    full = FluxAssetPolicy(make_tiny_policy(), _Autoencoder(), _PlainTextEncoder())
    assert full.checkpoint_contract() == {
        "history_pool_size": 4, "text_encoder": "_PlainTextEncoder",
    }
    wrapper = lora_wrapper()
    contract = wrapper.checkpoint_contract()
    assert contract["dit_lora"] == wrapper.parameter_report()["dit_lora"]
    assert contract["dit_lora"]["training_policy"] == DIT_LORA_TRAINING_POLICY
    assert lora_wrapper(dit_lora_rank=3).checkpoint_contract() != contract
    assert lora_wrapper(dit_lora_dropout=0.5).checkpoint_contract() != contract


def test_model_signature_separates_the_previous_whole_backbone_frozen_policy(monkeypatch):
    """A run recorded under the earlier freeze policy cannot resume silently."""
    from patchwam.engine import OptimizationRun

    wrapper = lora_wrapper()
    current = OptimizationRun._model_signature(wrapper)
    assert current["conditioning"]["dit_lora"]["training_policy"] == DIT_LORA_TRAINING_POLICY
    recorded = copy.deepcopy(current["conditioning"])
    del recorded["dit_lora"]["training_policy"]
    monkeypatch.setattr(wrapper, "checkpoint_contract", lambda: recorded)
    resumed = OptimizationRun._model_signature(wrapper)
    assert resumed["conditioning"] != current["conditioning"]
    assert json.dumps(resumed, sort_keys=True) != json.dumps(current, sort_keys=True)


def test_lora_checkpoints_roundtrip_and_reject_a_changed_adapter_layout(tmp_path):
    torch.manual_seed(9)
    source = lora_wrapper()
    path = tmp_path / "policy.safetensors"
    save_file(
        {name: tensor.detach().cpu().clone().contiguous() for name, tensor in policy_weight_state(source).items()},
        str(path),
    )
    destination = lora_wrapper()
    assert load_policy_weights(destination, path) == {"loaded_tensors": len(policy_weight_state(source))}
    stored = policy_weight_state(source)
    for name, tensor in policy_weight_state(destination).items():
        torch.testing.assert_close(tensor, stored[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="shape"):
        load_policy_weights(lora_wrapper(dit_lora_rank=3), path)
    with pytest.raises(ValueError, match="Incomplete"):
        load_policy_weights(FluxAssetPolicy(PatchFlowPolicy(
            OfficialFluxDenoiser(official_core()), action_dim=14, text_dim=32, proprio_dim=14,
        ), _Autoencoder(), _PlainTextEncoder()), path)


def test_engine_step_updates_adapters_and_writes_a_lora_checkpoint(tmp_path):
    from patchwam.engine import OptimizationRun, RunSettings
    from patchwam.testing import TensorExamples

    torch.manual_seed(7)
    model = lora_wrapper()
    before = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}
    run = OptimizationRun(
        model, TensorExamples(length=8),
        RunSettings(
            output_dir=str(tmp_path), epochs=1, max_updates=2, batch_size=2, accumulation=1,
            precision="no", cpu=True, workers=0, checkpoint_every=2, log_every=1, seed=5,
        ),
    )
    checkpoint = Path(run.train())
    assert run.updates == 2
    changed = {name for name, tensor in model.named_parameters() if not torch.equal(tensor, before[name])}
    assert any(name.endswith("residual_out.weight") for name in changed)
    assert all(DIT_PREFIX + name in changed for name in TRAINED_DIT)
    # Decoupled weight decay may also move the zero-gradient adapter input.
    assert all("residual" in name or name.startswith("policy.state_projection") or DIT_PREFIX in name
               for name in changed)
    assert not any(
        name.startswith(("autoencoder", "text_encoder")) or ".base." in name
        for name in changed
    )
    saved = load_file(str(checkpoint / "policy_0.safetensors"))
    assert any(name.endswith("residual_out.weight") for name in saved)
    live = dict(model.named_parameters())
    trained_in = next(name for name in saved if name.endswith("transformer.img_in.weight"))
    # The checkpoint stores the trained non-wrapped DiT weights and the frozen bases.
    assert torch.equal(saved[trained_in], live["policy." + trained_in])
    assert not torch.equal(saved[trained_in], before["policy." + trained_in])
    stored_base = next(name for name in saved if name.endswith("img_attn.qkv.base.weight"))
    assert torch.equal(saved[stored_base], before["policy." + stored_base])


def test_local_asset_factory_injects_after_loading_and_records_the_contract(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    ae_api = importlib.import_module("flux2.autoencoder")
    config = api.Klein4BParams(
        context_in_dim=24, hidden_size=64, num_heads=4, depth=1, depth_single_blocks=1,
        axes_dim=[4, 4, 4, 4], mlp_ratio=2,
    )
    ae_config = ae_api.AutoEncoderParams(resolution=16, ch=32, ch_mult=[1, 1], num_res_blocks=1)
    transformer, autoencoder = api.Flux2(config), ae_api.AutoEncoder(ae_config)
    transformer_path, ae_path = tmp_path / "transformer.safetensors", tmp_path / "ae.safetensors"
    save_file(transformer.state_dict(), str(transformer_path))
    save_file(autoencoder.state_dict(), str(ae_path))
    text_path = tmp_path / "text"
    transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=28,
        num_attention_heads=2, num_key_value_heads=2, head_dim=4,
    )).save_pretrained(text_path)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [entry[0]["content"] for entry in messages]

        def __call__(self, formatted, **kwargs):
            return transformers.BatchEncoding({
                "input_ids": torch.tensor([[1, 2, 3, 0, 0, 0]]).expand(len(formatted), -1),
                "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0]]).expand(len(formatted), -1),
            })

    monkeypatch.setattr(api, "Klein4BParams", lambda: config)
    monkeypatch.setattr(ae_api, "AutoEncoderParams", lambda: ae_config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    policy = FluxAssetPolicy.from_local_assets(
        str(transformer_path), str(ae_path), str(text_path), device="cpu", dtype="float32",
        context_length=6, gradient_checkpointing=True, dit_lora_rank=2, dit_lora_alpha=4,
        dit_lora_dropout=0, dit_lora=True,
    )
    loaded = {
        base_weight_name(name): parameter
        for name, parameter in policy.policy.denoiser.transformer.named_parameters()
    }
    for name, parameter in transformer.named_parameters():
        torch.testing.assert_close(loaded[base_weight_name(name)], parameter, rtol=0, atol=0)
    for name, parameter in policy.policy.denoiser.transformer.named_parameters():
        if "residual_out" in name:
            assert parameter.abs().sum() == 0
    report = policy.parameter_report()
    assert report["mode"] == "dit_lora" and report["dit_lora"]["rank"] == 2
    assert policy.checkpoint_contract()["dit_lora"]["adapter_targets"] == 10
    assert policy.checkpoint_contract()["dit_lora"]["training_policy"] == DIT_LORA_TRAINING_POLICY
    sample = {"video": torch.randn(2, 3, 2, 16, 16), "prompt": ["move", "lift"],
              "action": torch.randn(2, 16, 14), "proprio": torch.randn(2, 14)}
    policy.train()
    before = {name: tensor.detach().clone() for name, tensor in policy.named_parameters()}
    loss = policy(sample)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.policy.state_projection.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in policy.autoencoder.parameters())
    assert all(parameter.grad is None for parameter in policy.text_encoder.parameters())
    trained = dict(policy.named_parameters())
    assert all(trained[DIT_PREFIX + name].grad is not None for name in (
        "img_in.weight", "time_in.in_layer.weight", "final_layer.linear.weight",
        "single_stream_modulation.lin.weight", "double_blocks.0.img_attn.norm.query_norm.scale",
    ))
    assert all(trained[DIT_PREFIX + name].grad is None for name in (
        "double_blocks.0.img_attn.qkv.base.weight", "single_blocks.0.linear2.base.weight",
    ))
    torch.optim.SGD(
        [parameter for parameter in policy.parameters() if parameter.requires_grad], lr=1e-3,
    ).step()
    changed = {name for name, tensor in policy.named_parameters() if not torch.equal(tensor, before[name])}
    assert all(DIT_PREFIX + name in changed for name in TRAINED_DIT)
    assert not any(".base." in name for name in changed)
    assert not any(name.startswith(("autoencoder", "text_encoder")) for name in changed)


def test_local_asset_factory_rejects_trainable_vl_adapters_with_dit_lora(tmp_path):
    with pytest.raises(ValueError, match="vl_trainable_adapters"):
        FluxAssetPolicy.from_local_assets(
            "missing.safetensors", "missing.safetensors", str(tmp_path), device="cpu",
            dit_lora=True, vl_trainable_adapters=True,
        )
    with pytest.raises(ValueError, match="dit_lora_rank"):
        FluxAssetPolicy.from_local_assets(
            "missing.safetensors", "missing.safetensors", str(tmp_path), device="cpu",
            dit_lora=True, dit_lora_rank=0,
        )
    with pytest.raises(ValueError, match="dit_lora must be a boolean"):
        FluxAssetPolicy.from_local_assets(
            "missing.safetensors", "missing.safetensors", str(tmp_path), device="cpu", dit_lora="yes",
        )


def test_training_command_records_the_parameter_report_for_a_lora_run(tmp_path, monkeypatch, capsys):
    transformers = pytest.importorskip("transformers")
    ae_api = importlib.import_module("flux2.autoencoder")
    config = api.Klein4BParams(
        context_in_dim=24, hidden_size=64, num_heads=4, depth=1, depth_single_blocks=1,
        axes_dim=[4, 4, 4, 4], mlp_ratio=2,
    )
    ae_config = ae_api.AutoEncoderParams(resolution=16, ch=32, ch_mult=[1, 1], num_res_blocks=1)
    transformer, autoencoder = api.Flux2(config), ae_api.AutoEncoder(ae_config)
    transformer_path, ae_path = tmp_path / "transformer.safetensors", tmp_path / "ae.safetensors"
    save_file(transformer.state_dict(), str(transformer_path))
    save_file(autoencoder.state_dict(), str(ae_path))
    text_path = tmp_path / "text"
    transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=28,
        num_attention_heads=2, num_key_value_heads=2, head_dim=4,
    )).save_pretrained(text_path)

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            raise AssertionError("Cached text tokens must not tokenize prompts")

    monkeypatch.setattr(api, "Klein4BParams", lambda: config)
    monkeypatch.setattr(ae_api, "AutoEncoderParams", lambda: ae_config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    output = tmp_path / "run"
    (tmp_path / "lora.yaml").write_text(
        f"""model:
  _target_: patchwam.models.FluxAssetPolicy.from_local_assets
  transformer_path: {transformer_path}
  autoencoder_path: {ae_path}
  text_encoder_path: {text_path}
  action_dim: 14
  proprio_dim: 14
  variant: klein-base-4b
  dtype: float32
  device: cpu
  context_length: 6
  dit_lora: true
  dit_lora_rank: 2
  dit_lora_alpha: 4
  dit_lora_dropout: 0
data:
  _target_: patchwam.testing.TensorExamples
  length: 4
  text_dim: 24
training:
  output_dir: {output}
  epochs: 1
  max_updates: 1
  batch_size: 2
  accumulation: 1
  precision: "no"
  cpu: true
  workers: 0
  checkpoint_every: 1
  log_every: 1
  seed: 3
  global_batch_size: 2
"""
    )
    from patchwam.cli import main

    monkeypatch.setattr("sys.argv", ["patchwam", "train", "--config", str(tmp_path / "lora.yaml")])
    main()
    report = json.loads((output / "parameter_report.json").read_text())
    assert report["mode"] == "dit_lora"
    assert report["dit_lora"]["training_policy"] == DIT_LORA_TRAINING_POLICY
    # 10 wrapped projections, 15 remaining DiT tensors, and the state projection.
    assert report["counts"]["adapters"] == 20
    assert report["counts"]["fully_trainable"] == 17
    assert report["counts"]["trainable"] == 37
    assert report["counts"]["frozen"] > 37  # the real autoencoder and Qwen3 encoder
    assert report["counts"]["total"] == report["counts"]["trainable"] + report["counts"]["frozen"]
    assert len(report["frozen"]["groups"]["policy.denoiser.transformer"]["layers"]) == 10
    assert (output / "configuration.yaml").exists()
    logged = [line for line in capsys.readouterr().out.splitlines() if "parameter_report" in line]
    assert logged and json.loads(logged[0])["mode"] == "dit_lora"
