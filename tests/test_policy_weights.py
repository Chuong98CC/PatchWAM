"""Native policy initialization and frozen-asset checkpoint coverage."""

import copy
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from patchwam.checkpoints import load_policy_weights, register_compact_policy_state
from patchwam.models import make_tiny_policy
from patchwam.models.lora import LowRankResidualLinear, attach_block_adapters


def tensor_only_copy(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def weight_copy(policy):
    return {key: value.detach().clone().contiguous() for key, value in policy.state_dict().items()}


@pytest.mark.parametrize("wrapped", [False, True])
def test_native_policy_weights_load_completely(tmp_path, wrapped):
    source, destination = make_tiny_policy(), make_tiny_policy()
    path = tmp_path / "policy.safetensors"
    save_file(weight_copy(source), str(path))
    model = SimpleNamespace(policy=destination) if wrapped else destination
    assert load_policy_weights(model, path) == {"loaded_tensors": len(source.state_dict())}
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["shape", "missing", "unexpected", "nonfinite", "integer", "overflow"])
def test_invalid_policy_weights_do_not_change_any_parameters(tmp_path, damage):
    source, destination = make_tiny_policy(), make_tiny_policy()
    before = weight_copy(destination)
    weights = weight_copy(source)
    key = next(iter(weights))
    if damage == "shape":
        weights[key] = torch.ones(1)
    elif damage == "missing":
        del weights[key]
    elif damage == "unexpected":
        weights["unrecognized.weight"] = torch.ones(1)
    elif damage == "nonfinite":
        weights[key].flatten()[0] = float("nan")
    elif damage == "integer":
        weights[key] = weights[key].long()
    else:
        weights[key] = torch.full_like(weights[key], 1e100, dtype=torch.float64)
    path = tmp_path / "invalid.safetensors"
    save_file(weights, str(path))
    with pytest.raises(ValueError):
        load_policy_weights(destination, path)
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("extension", [".bin", ".ckpt"])
def test_policy_initialization_rejects_unsupported_formats(tmp_path, extension):
    """Tensor-only .pt/.pth files are supported; other formats stay rejected."""
    with pytest.raises(ValueError, match="native .safetensors"):
        load_policy_weights(make_tiny_policy(), tmp_path / ("weights" + extension))


def test_missing_policy_weight_file_reports_its_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policy_weights(make_tiny_policy(), tmp_path / "weights.pt")


@pytest.mark.parametrize("suffix", [".pt", ".pth"])
def test_tensor_only_checkpoint_initializes_the_complete_policy(tmp_path, suffix):
    source, destination = make_tiny_policy(), make_tiny_policy()
    path = tmp_path / ("policy" + suffix)
    torch.save(tensor_only_copy(source), path)
    assert load_policy_weights(destination, path) == {"loaded_tensors": len(source.state_dict())}
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("damage, message", [
    ("unsafe", "weights_only"),
    ("missing", "Incomplete"),
    ("unexpected", "Incomplete"),
])
def test_tensor_only_checkpoints_are_rejected_atomically(tmp_path, damage, message):
    source, destination = make_tiny_policy(), make_tiny_policy()
    before = weight_copy(destination)
    weights = tensor_only_copy(source)
    if damage == "unsafe":
        payload = {"policy": nn.Linear(2, 2)}
    elif damage == "missing":
        del weights[next(iter(weights))]
        payload = weights
    else:
        payload = weights | {"unrecognized.weight": torch.ones(1)}
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        load_policy_weights(destination, path)
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


class _ContractBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.linear1 = nn.Linear(width, width * 2, bias=False)
        self.linear2 = nn.Linear(width * 2, width, bias=False)


class ContractTransformer(nn.Module):
    """The official block contract: projections named qkv/proj/linear1/linear2."""

    def __init__(self, width=8):
        super().__init__()
        self.double_blocks = nn.ModuleList([_ContractBlock(width)])
        self.single_blocks = nn.ModuleList([_ContractBlock(width)])
        self.img_in = nn.Linear(width, width, bias=False)


def test_tensor_only_base_weights_fill_lora_projections_and_leave_adapters_fresh(tmp_path):
    """A frozen-base checkpoint initializes the PEFT LORA projections on the local CPU."""
    torch.manual_seed(11)
    plain, destination = ContractTransformer(), ContractTransformer()
    attach_block_adapters(destination, rank=2, alpha=4, dropout=0)
    adapters = {
        name: value.detach().clone()
        for name, value in destination.state_dict().items()
        if ".lora_A." in name
    }
    assert adapters
    path = tmp_path / "base.pt"
    torch.save(tensor_only_copy(plain), path)
    assert load_policy_weights(destination, path) == {"loaded_tensors": len(plain.state_dict())}
    loaded = destination.state_dict()
    for name, value in plain.state_dict().items():
        wrapped = name[: -len("weight")] + "base_layer.weight" if name.endswith(".weight") else name
        target = wrapped if wrapped in loaded else name
        torch.testing.assert_close(loaded[target], value, rtol=0, atol=0)
    # Block projections moved behind their frozen base layer; the rest stayed put.
    assert "double_blocks.0.qkv.weight" not in loaded and "img_in.base_layer.weight" not in loaded
    for name, value in adapters.items():
        torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)
    zeros = [name for name in loaded if name.endswith("lora_B.default.weight")]
    assert zeros and all(loaded[name].abs().sum() == 0 for name in zeros)


def test_legacy_custom_lora_tensors_initialize_the_peft_adapter(tmp_path):
    """A checkpoint from the custom DiT LoRA path converts without losing tensors."""
    torch.manual_seed(13)
    plain = ContractTransformer()
    legacy = copy.deepcopy(plain)
    wrapped = {}
    for name, module in list(legacy.named_modules()):
        if isinstance(module, nn.Linear) and name.split(".")[0] in {"double_blocks", "single_blocks"}:
            parent_name, _, attribute = name.rpartition(".")
            parent = legacy.get_submodule(parent_name)
            adapter = LowRankResidualLinear(module, rank=2, alpha=4, dropout=0)
            with torch.no_grad():
                adapter.residual_in.weight.fill_(0.25)
                adapter.residual_out.weight.fill_(0.5)
            setattr(parent, attribute, adapter)
            wrapped[name] = adapter
    assert wrapped
    path = tmp_path / "legacy.safetensors"
    save_file(tensor_only_copy(legacy), str(path))

    destination = ContractTransformer()
    attach_block_adapters(destination, rank=2, alpha=4, dropout=0)
    assert load_policy_weights(destination, path) == {"loaded_tensors": len(legacy.state_dict())}
    converted = destination.state_dict()
    for name, adapter in wrapped.items():
        torch.testing.assert_close(
            converted[f"{name}.lora_A.default.weight"], adapter.residual_in.weight, rtol=0, atol=0,
        )
        torch.testing.assert_close(
            converted[f"{name}.lora_B.default.weight"], adapter.residual_out.weight, rtol=0, atol=0,
        )
        torch.testing.assert_close(
            converted[f"{name}.base_layer.weight"], adapter.base.weight, rtol=0, atol=0,
        )


def test_legacy_custom_lora_tensors_are_rejected_when_no_peft_adapter_matches(tmp_path):
    """Adapter tensors are never dropped silently when the target layout differs."""
    torch.manual_seed(17)
    legacy = ContractTransformer()
    legacy.double_blocks[0].qkv = LowRankResidualLinear(legacy.double_blocks[0].qkv, rank=2, alpha=4, dropout=0)
    path = tmp_path / "legacy.safetensors"
    save_file(tensor_only_copy(legacy), str(path))
    plain_destination = ContractTransformer()
    before = tensor_only_copy(plain_destination)
    with pytest.raises(ValueError, match="custom DiT LoRA adapter tensors"):
        load_policy_weights(plain_destination, path)
    for name, value in plain_destination.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    # A different rank at the same projection is a shape conflict, not a silent skip.
    mismatched = ContractTransformer()
    attach_block_adapters(mismatched, rank=3, alpha=4, dropout=0)
    with pytest.raises(ValueError, match="shape"):
        load_policy_weights(mismatched, path)


def test_native_peft_ia3_checkpoint_round_trips_without_a_wrapper(tmp_path):
    """IA3 tensors save and load exactly through the policy state dictionary."""
    from patchwam.models.lora import attach_adapter_layers, build_dit_adapter_config

    torch.manual_seed(23)
    source, destination = ContractTransformer(), ContractTransformer()
    for model in (source, destination):
        attach_adapter_layers(model, build_dit_adapter_config(model, {"peft_type": "IA3"}))
    assert any(name.endswith(".ia3_l.default") for name in source.state_dict())
    # Low-level injection keeps the policy state dictionary flat: no PeftModel
    # wrapper and no duplicated base-model parameters.
    assert not any("base_model" in name for name in source.state_dict())
    with torch.no_grad():
        source.double_blocks[0].qkv.ia3_l["default"].fill_(1.5)
    path = tmp_path / "ia3.safetensors"
    save_file(tensor_only_copy(source), str(path))
    assert load_policy_weights(destination, path) == {"loaded_tensors": len(source.state_dict())}
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[name], rtol=0, atol=0)
    assert destination.double_blocks[0].qkv.ia3_l["default"].flatten()[0].item() == 1.5


class HookHarness:
    is_main_process = True

    @staticmethod
    def unwrap_model(model):
        return model

    def register_save_state_pre_hook(self, hook):
        self.save = hook

    def register_load_state_pre_hook(self, hook):
        self.load = hook


def test_compact_hook_roundtrip_excludes_frozen_encoder_weights(tmp_path):
    policy = make_tiny_policy()
    wrapper = SimpleNamespace(policy=policy, autoencoder=nn.Linear(4, 4), text_encoder=nn.Linear(4, 4))
    accelerator = HookHarness()
    register_compact_policy_state(accelerator)
    expected = weight_copy(policy)
    asset_before = wrapper.autoencoder.weight.detach().clone()
    weights = [{"sentinel": torch.tensor(1)}]
    accelerator.save([wrapper], weights, str(tmp_path))
    assert weights == []
    saved = load_file(str(tmp_path / "policy_0.safetensors"))
    assert saved.keys() == expected.keys()
    for name, tensor in saved.items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    models = [wrapper]
    accelerator.load(models, str(tmp_path))
    assert models == []
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    torch.testing.assert_close(wrapper.autoencoder.weight, asset_before, rtol=0, atol=0)


def test_asset_signature_detects_weight_metadata_and_small_source_content_changes(tmp_path):
    import os

    from patchwam.models.flux import _asset_identity

    transformer = tmp_path / "transformer.safetensors"
    autoencoder = tmp_path / "autoencoder.safetensors"
    source = tmp_path / "model.py"
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    transformer.write_bytes(b"weights")
    autoencoder.write_bytes(b"encoder")
    source.write_text("value = 1")
    (text_dir / "config.json").write_text('{"hidden_size": 32}')
    modules = (SimpleNamespace(__file__=str(source)),)
    first = _asset_identity(transformer, autoencoder, text_dir, modules)
    original_time = source.stat().st_mtime_ns
    source.write_text("value = 2")
    os.utime(source, ns=(original_time, original_time))
    second = _asset_identity(transformer, autoencoder, text_dir, modules)
    assert first["source"][0]["size"] == second["source"][0]["size"]
    assert first["source"][0]["mtime_ns"] == second["source"][0]["mtime_ns"]
    assert first != second
    autoencoder.write_bytes(b"different encoder")
    assert second != _asset_identity(transformer, autoencoder, text_dir, modules)


def test_explicit_backbone_source_rejects_previously_imported_other_tree(tmp_path, monkeypatch):
    import sys

    from patchwam.models import FluxAssetPolicy, flux

    transformer = tmp_path / "transformer.safetensors"
    autoencoder = tmp_path / "autoencoder.safetensors"
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    transformer.write_bytes(b"weights")
    autoencoder.write_bytes(b"weights")
    source_root = tmp_path / "requested"
    package = source_root / "src" / "flux2"
    package.mkdir(parents=True)
    (package / "model.py").write_text("")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(flux.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(tmp_path / "other.py")))
    with pytest.raises(RuntimeError, match="Another FLUX.2 source"):
        FluxAssetPolicy.from_local_assets(str(transformer), str(autoencoder), str(text_dir),
                                         flux_source=str(source_root), device="cpu")
