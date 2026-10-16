"""Tensor-only PyTorch checkpoints, upstream mapping, and PEFT initialization."""

import copy
import os
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from patchwam.checkpoints import load_policy_weights, policy_weight_state
from patchwam.models import FluxAssetPolicy, joint_visibility, sequence_coordinates
from patchwam.models.flux import OfficialFluxDenoiser
from patchwam.models.lora import is_adapter_layer
from patchwam.models.policy import PatchFlowPolicy, make_tiny_policy

REPOSITORY = Path(__file__).resolve().parents[1]
RELEASE_CHECKPOINT = Path(
    os.environ.get("PATCHWAM_RELEASE_CHECKPOINT", REPOSITORY / "Pretrained_models/patchwam-robotwin/step_140000.pt")
)
TRANSFORMER_PREFIX = "denoiser.transformer."
RELEASE_PREFIX = "mixtures.video.transformer."
RELEASE_ALIAS_ROOT = "mixtures.video."

api = pytest.importorskip("flux2.model", reason="Install the official FLUX.2 dependency for release-layout tests")


def official_core():
    return api.Flux2(api.Klein4BParams(
        context_in_dim=32, hidden_size=64, num_heads=4, depth=2, depth_single_blocks=2,
        axes_dim=[4, 4, 4, 4], mlp_ratio=2,
    ))


def patch_policy(core=None, **changes):
    return PatchFlowPolicy(
        OfficialFluxDenoiser(official_core() if core is None else core, **changes),
        action_dim=14, text_dim=32, proprio_dim=14,
    )


class _Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 128, 1)

    def encode(self, images):
        return F.adaptive_avg_pool2d(self.projection(images), (2, 2))


class _PlainTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(3, 32))

    def forward(self, prompts):
        return self.embeddings.unsqueeze(0).expand(len(prompts), -1, -1)


def release_payload(source, *, aliases=True, alias_copies=False):
    """An upstream PatchWAM release layout built from a pretrained policy."""
    official = source.denoiser.transformer.state_dict()
    mot = OrderedDict((RELEASE_PREFIX + name, tensor) for name, tensor in official.items())
    if aliases:
        for name, tensor in official.items():
            if name.startswith(("double_blocks.", "single_blocks.")):
                mot[RELEASE_ALIAS_ROOT + name] = tensor.clone() if alias_copies else tensor
    return {
        "mot": mot,
        "step": 140000,
        "torch_dtype": "torch.float32",
        "proprio_encoder": OrderedDict(source.state_projection.state_dict()),
    }


def write_release(path, source, **changes):
    payload = release_payload(source, **changes)
    torch.save(payload, path)
    return path


def base_name(name):
    """The PEFT ``base_layer`` name of a projection inside an adapted denoiser."""
    return name[: -len("weight")] + "base_layer.weight" if name.endswith(".weight") else name


def snapshot(module):
    return {name: tensor.detach().clone() for name, tensor in policy_weight_state(module).items()}


def assert_unchanged(module, before):
    for name, tensor in policy_weight_state(module).items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_release_layout_loads_every_pretrained_weight(tmp_path):
    source, destination = patch_policy(), patch_policy()
    payload = release_payload(source)
    assert len(payload["mot"]) > len(source.denoiser.transformer.state_dict())
    path = tmp_path / "step_140000.pt"
    torch.save(payload, path)
    assert load_policy_weights(destination, path) == {"loaded_tensors": len(source.state_dict())}
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(destination.state_dict()[name], tensor, rtol=0, atol=0)


def test_release_layout_accepts_repeated_block_weights_stored_separately(tmp_path):
    source = patch_policy()
    mot_source = release_payload(source, alias_copies=True)
    aliases = [key for key in mot_source["mot"] if key.startswith("mixtures.video.") and
               not key.startswith("mixtures.video.transformer.")]
    assert aliases
    assert all(
        mot_source["mot"][key].untyped_storage().data_ptr()
        != mot_source["mot"][RELEASE_PREFIX + key[len(RELEASE_ALIAS_ROOT):]].untyped_storage().data_ptr()
        for key in aliases
    )
    path = tmp_path / "copies.pt"
    torch.save(mot_source, path)
    destination = patch_policy()
    load_policy_weights(destination, path)
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(destination.state_dict()[name], tensor, rtol=0, atol=0)


def test_release_layout_leaves_lora_adapters_fresh_and_zero(tmp_path):
    source = patch_policy()
    path = write_release(tmp_path / "step_140000.pt", source)
    destination = patch_policy(dit_lora=True, dit_lora_rank=2, dit_lora_alpha=4, dit_lora_dropout=0)
    adapters = {
        name: parameter.detach().clone()
        for name, parameter in destination.denoiser.transformer.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    assert adapters
    load_policy_weights(destination, path)
    loaded = dict(destination.denoiser.transformer.named_parameters())
    for name, tensor in source.denoiser.transformer.state_dict().items():
        target = base_name(name) if base_name(name) in loaded else name
        torch.testing.assert_close(loaded[target], tensor, rtol=0, atol=0)
    zeros = [name for name in loaded if name.endswith("lora_B.default.weight")]
    assert len(zeros) == destination.denoiser.dit_lora_contract()["adapter_targets"]
    assert all(loaded[name].abs().sum() == 0 for name in zeros)
    for name, tensor in adapters.items():
        torch.testing.assert_close(loaded[name], tensor, rtol=0, atol=0)
    torch.testing.assert_close(
        destination.state_projection.weight, source.state_projection.weight, rtol=0, atol=0,
    )


def test_fresh_lora_adapters_preserve_the_pretrained_denoiser_output(tmp_path):
    torch.manual_seed(4)
    source = patch_policy()
    destination = patch_policy(copy.deepcopy(source.denoiser.transformer),
                               dit_lora=True, dit_lora_rank=2, dit_lora_alpha=4, dit_lora_dropout=0)
    plain = patch_policy(source.denoiser.transformer)
    data = {
        "reference": torch.randn(2, 3, 128), "noisy": torch.randn(2, 6, 128),
        "context": torch.randn(2, 4, 32), "sigma": torch.tensor([0.3, 0.7]),
        "reference_ids": sequence_coordinates(2, 3, group=10), "noisy_ids": sequence_coordinates(2, 6, group=20),
        "context_ids": sequence_coordinates(2, 4), "visibility": joint_visibility(4, 3, 4, 2),
    }
    with torch.no_grad():
        # PEFT zero-initializes the LoRA up-projection, so a fresh adapter adds an
        # exact zero residual; only the float32 GEMM rounding of the frozen base
        # layers remains, as in the adapter tests.
        torch.testing.assert_close(destination.denoiser(**data), plain.denoiser(**data), rtol=1e-5, atol=1e-6)
        for module in destination.denoiser.transformer.modules():
            if is_adapter_layer(module):
                base = module.get_base_layer()
                probe = torch.randn(3, base.in_features)
                torch.testing.assert_close(module(probe), base(probe), rtol=0, atol=0)


def test_release_layout_initializes_the_wrapped_policy_and_leaves_frozen_assets(tmp_path):
    source = patch_policy()
    path = write_release(tmp_path / "step_140000.pt", source)
    wrapper = FluxAssetPolicy(patch_policy(), _Autoencoder(), _PlainTextEncoder())
    assets = {
        name: tensor.detach().clone() for name, tensor in wrapper.state_dict().items()
        if name.startswith(("autoencoder.", "text_encoder."))
    }
    load_policy_weights(wrapper, path)
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(wrapper.policy.state_dict()[name], tensor, rtol=0, atol=0)
    for name, tensor in wrapper.state_dict().items():
        if name in assets:
            torch.testing.assert_close(tensor, assets[name], rtol=0, atol=0)


@pytest.mark.parametrize("suffix", [".pt", ".pth"])
def test_native_state_dictionary_round_trips_through_torch_save(tmp_path, suffix):
    source, destination = make_tiny_policy(), make_tiny_policy()
    path = tmp_path / ("policy" + suffix)
    torch.save({name: tensor.detach().clone() for name, tensor in source.state_dict().items()}, path)
    assert load_policy_weights(destination, path) == {"loaded_tensors": len(source.state_dict())}
    for name, tensor in destination.state_dict().items():
        torch.testing.assert_close(tensor, source.state_dict()[name], rtol=0, atol=0)


def test_safetensors_initialization_still_requires_the_complete_schema(tmp_path):
    source, destination = make_tiny_policy(), make_tiny_policy()
    path = tmp_path / "policy.safetensors"
    save_file({name: tensor.detach().clone() for name, tensor in source.state_dict().items()}, str(path))
    assert load_policy_weights(destination, path) == {"loaded_tensors": len(source.state_dict())}


def test_pickled_objects_are_rejected_without_an_unsafe_fallback(tmp_path):
    destination = make_tiny_policy()
    before = snapshot(destination)
    path = tmp_path / "module.pt"
    torch.save({"model": nn.Linear(2, 2)}, path)
    with pytest.raises(ValueError, match="weights_only"):
        load_policy_weights(destination, path)
    assert_unchanged(destination, before)


@pytest.mark.parametrize("payload", [
    [torch.zeros(2)],
    {"state_dict": {"weight": torch.zeros(1)}},
    {"mot": {"mixtures.video.transformer.img_in.weight": torch.zeros(1)}},
])
def test_unsupported_torch_layouts_are_rejected(tmp_path, payload):
    destination = make_tiny_policy()
    before = snapshot(destination)
    path = tmp_path / "unsupported.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unsupported|state dictionary"):
        load_policy_weights(destination, path)
    assert_unchanged(destination, before)


def test_unsupported_extension_and_missing_file_are_reported(tmp_path):
    with pytest.raises(ValueError, match="native .safetensors"):
        load_policy_weights(make_tiny_policy(), tmp_path / "weights.bin")
    with pytest.raises(FileNotFoundError):
        load_policy_weights(make_tiny_policy(), tmp_path / "weights.pt")


def damaged_release(tmp_path, source, change):
    payload = release_payload(source)
    name = "mixtures.video.transformer.img_in.weight"
    if change == "missing":
        del payload["mot"][name]
    elif change == "shape":
        payload["mot"][name] = torch.ones(1)
    elif change == "nonfinite":
        payload["mot"][name] = torch.full_like(payload["mot"][name], float("nan"))
    elif change == "conflicting-alias":
        alias = "mixtures.video.double_blocks.0.img_attn.qkv.weight"
        payload["mot"][alias] = payload["mot"][alias].clone() + 1
    elif change == "orphan-alias":
        alias = "mixtures.video.double_blocks.0.img_attn.qkv.weight"
        transformer = "mixtures.video.transformer.double_blocks.0.img_attn.qkv.weight"
        payload["mot"][alias] = payload["mot"].pop(transformer)
    elif change == "unknown-entry":
        payload["mot"]["mixtures.videoother.weight"] = torch.ones(1)
    else:
        raise AssertionError(change)
    path = tmp_path / f"{change}.pt"
    torch.save(payload, path)
    return path


@pytest.mark.parametrize("change, message", [
    ("missing", "Incomplete"),
    ("shape", "shape"),
    ("nonfinite", "invalid"),
    ("conflicting-alias", "conflicting"),
    ("orphan-alias", "without its"),
    ("unknown-entry", "unsupported 'mot' entries"),
])
def test_incompatible_release_checkpoints_are_rejected_before_mutation(tmp_path, change, message):
    source, destination = patch_policy(), patch_policy()
    before = snapshot(destination)
    path = damaged_release(tmp_path, source, change)
    with pytest.raises(ValueError, match=message):
        load_policy_weights(destination, path)
    assert_unchanged(destination, before)


def test_release_payload_without_the_state_projection_is_rejected(tmp_path):
    source = patch_policy()
    payload = release_payload(source)
    del payload["proprio_encoder"]
    path = tmp_path / "no-projection.pt"
    torch.save(payload, path)
    destination = patch_policy()
    before = snapshot(destination)
    with pytest.raises(ValueError, match="proprio_encoder"):
        load_policy_weights(destination, path)
    assert_unchanged(destination, before)


@pytest.mark.skipif(not RELEASE_CHECKPOINT.is_file(), reason="Local release checkpoint is not available")
def test_actual_release_checkpoint_maps_onto_the_official_transformer():
    """The shipped step_140000.pt initializes the Klein base 4B policy on CPU."""
    with torch.device("meta"):
        core = api.Flux2(api.Klein4BParams())
    model = PatchFlowPolicy(OfficialFluxDenoiser(core), action_dim=14, text_dim=7680, proprio_dim=14)
    with pytest.warns(UserWarning, match="meta parameter"):
        assert load_policy_weights(model, RELEASE_CHECKPOINT) == {"loaded_tensors": 151}
    reference = torch.load(str(RELEASE_CHECKPOINT), map_location="cpu", weights_only=True, mmap=True)
    assert set(reference) == {"mot", "step", "torch_dtype", "proprio_encoder"}
    assert reference["step"] == 140000 and reference["torch_dtype"] == "torch.bfloat16"
    transformer = {key[len("mixtures.video.transformer."):]: value for key, value in reference["mot"].items()
                   if key.startswith("mixtures.video.transformer.")}
    assert set(transformer) == set(model.denoiser.transformer.state_dict())
    for name, parameter in model.denoiser.transformer.state_dict().items():
        assert transformer[name].shape == parameter.shape
    assert reference["proprio_encoder"]["weight"].shape == model.state_projection.weight.shape
    torch.testing.assert_close(
        model.state_projection.bias, reference["proprio_encoder"]["bias"].to(torch.float32), rtol=0, atol=0,
    )
    aliases = {key[len("mixtures.video."):]: value for key, value in reference["mot"].items()
               if key.startswith(("mixtures.video.double_blocks.", "mixtures.video.single_blocks."))}
    assert len(aliases) == 140
    assert all(
        value.untyped_storage().data_ptr() == transformer[name].untyped_storage().data_ptr()
        for name, value in aliases.items()
    )
