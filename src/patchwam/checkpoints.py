"""Policy weights and compact training checkpoints."""

from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .models.lora import adapter_parameter_name

# Upstream PatchWAM release checkpoints keep the official FLUX.2 transformer
# under ``mixtures.video.transformer``, repeat the block weights under a bare
# ``mixtures.video`` prefix that shares their storage, and store the proprio
# state projection as ``proprio_encoder``.
UPSTREAM_TRANSFORMER_PREFIX = "mixtures.video.transformer."
UPSTREAM_ALIAS_ROOT = "mixtures.video."
UPSTREAM_ALIAS_PREFIXES = (UPSTREAM_ALIAS_ROOT + "double_blocks.", UPSTREAM_ALIAS_ROOT + "single_blocks.")
UPSTREAM_STATE_PROJECTION = "proprio_encoder"

TRANSFORMER_KEY_PREFIX = "denoiser.transformer."
STATE_PROJECTION_KEY_PREFIX = "state_projection."
POLICY_KEY_PREFIX = "policy."

NATIVE_SUFFIX = ".safetensors"
TORCH_SUFFIXES = (".pt", ".pth")


def _policy_layout(model):
    policy = getattr(model, "policy", model)
    named = getattr(model, "named_parameters", None)
    extra = (
        {}
        if policy is model or named is None
        else {
            name: tensor
            for name, tensor in named()
            if tensor.requires_grad and not name.startswith("policy.")
        }
    )
    return policy, extra


def policy_weight_state(model):
    """Collect policy state and trainable encoder adapters."""
    policy, extra = _policy_layout(model)
    state = policy.state_dict()
    if extra:
        state = {"policy." + key: tensor for key, tensor in state.items()}
        state.update({key: tensor.detach() for key, tensor in extra.items()})
    return state


def checkpoint_parameter_keys(model):
    """Map model parameters to checkpoint keys."""
    policy, extra = _policy_layout(model)
    target = model if hasattr(model, "named_parameters") else policy
    result = {}
    for name, tensor in target.named_parameters(remove_duplicate=False):
        if policy is model or target is policy:
            result[name] = name
        elif name.startswith("policy."):
            result[name] = name if extra else name.removeprefix("policy.")
        elif name in extra:
            result[name] = name
    return result


def _unreadable(path, error):
    """Report an unreadable PyTorch file without offering a pickle fallback."""
    return ValueError(
        f"Could not read {path} as a tensor-only PyTorch state dictionary "
        f"with torch.load(weights_only=True): {type(error).__name__}: {error}. "
        "PatchWAM never falls back to unsafe pickle deserialization; re-export the "
        "weights as native .safetensors or as a tensor-only .pt/.pth state dictionary."
    )


def _torch_state_dictionary(path):
    """Read a PyTorch state file on CPU with weights-only deserialization."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise _unreadable(path, error) from error
    except Exception as error:  # noqa: BLE001 - report every unsafe or unreadable file the same way
        raise _unreadable(path, error) from error
    # Legacy non-zip files cannot be memory mapped; the same safe loader still applies.
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001
        raise _unreadable(path, error) from error


def _same_storage(first, second):
    return (
        first.shape == second.shape
        and first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
        and first.storage_offset() == second.storage_offset()
    )


def _upstream_alias_name(key):
    for prefix in UPSTREAM_ALIAS_PREFIXES:
        if key.startswith(prefix):
            return key[len(UPSTREAM_ALIAS_ROOT):]
    return None


def _upstream_tensors(loaded, path):
    """Map an upstream release checkpoint onto native policy weight names."""
    if not isinstance(loaded["mot"], Mapping):
        raise ValueError(f"{path} has an unsupported layout: 'mot' must be a state dictionary")
    values = loaded.get(UPSTREAM_STATE_PROJECTION)
    if values is None:
        raise ValueError(
            f"{path} uses an unsupported upstream layout: 'mot' requires a "
            f"'{UPSTREAM_STATE_PROJECTION}' state dictionary containing 'weight' and 'bias'"
        )
    if not isinstance(values, Mapping):
        raise ValueError(f"{path} has an unsupported layout: '{UPSTREAM_STATE_PROJECTION}' must be a state dictionary")
    step = loaded.get("step")
    if step is not None and (not isinstance(step, int) or isinstance(step, bool) or step < 0):
        raise ValueError(f"{path} has an unsupported layout: 'step' must be a nonnegative integer")
    dtype = loaded.get("torch_dtype")
    if dtype is not None and not isinstance(dtype, str):
        raise ValueError(f"{path} has an unsupported layout: 'torch_dtype' must be a string")
    tensors, aliases, unexpected = {}, {}, []
    for key, value in loaded["mot"].items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError(f"{path} has an unsupported layout: 'mot' must map names to tensors, found {key!r}")
        if key.startswith(UPSTREAM_TRANSFORMER_PREFIX):
            tensors[TRANSFORMER_KEY_PREFIX + key[len(UPSTREAM_TRANSFORMER_PREFIX):]] = value
            continue
        alias = _upstream_alias_name(key)
        if alias is None:
            unexpected.append(key)
        else:
            aliases[alias] = value
    if unexpected:
        raise ValueError(
            f"{path} has unsupported 'mot' entries {sorted(unexpected)[:10]}; expected names under "
            f"'{UPSTREAM_TRANSFORMER_PREFIX}' or its {' and '.join(UPSTREAM_ALIAS_PREFIXES)} aliases"
        )
    if not tensors:
        raise ValueError(f"{path} contains no '{UPSTREAM_TRANSFORMER_PREFIX}' transformer weights")
    for alias, value in aliases.items():
        name = TRANSFORMER_KEY_PREFIX + alias
        original = tensors.get(name)
        if original is None:
            raise ValueError(
                f"{path} repeats the alias weight {alias} without its "
                f"'{UPSTREAM_TRANSFORMER_PREFIX}{alias}' transformer weight"
            )
        if value.shape != original.shape or value.dtype != original.dtype:
            raise ValueError(
                f"{path} stores the repeated weight {alias} with a different shape or dtype "
                f"than '{UPSTREAM_TRANSFORMER_PREFIX}{alias}'"
            )
        if not _same_storage(value, original) and not torch.equal(value, original):
            raise ValueError(
                f"{path} stores conflicting values for the repeated weight {alias} and "
                f"'{UPSTREAM_TRANSFORMER_PREFIX}{alias}'"
            )
    for leaf in ("weight", "bias"):
        value = values.get(leaf)
        if not torch.is_tensor(value):
            raise ValueError(
                f"{path} has an unsupported layout: '{UPSTREAM_STATE_PROJECTION}' needs tensor "
                f"'{leaf}' entries, found {type(value).__name__}"
            )
        tensors[STATE_PROJECTION_KEY_PREFIX + leaf] = value
    return tensors


def _read_policy_weights(path):
    """Return native policy-space tensors from every supported checkpoint layout."""
    if path.suffix == NATIVE_SUFFIX:
        return dict(load_file(str(path)))
    if path.suffix not in TORCH_SUFFIXES:
        raise ValueError(
            "Policy weights must use native .safetensors or a tensor-only .pt/.pth state dictionary"
        )
    loaded = _torch_state_dictionary(path)
    if not isinstance(loaded, Mapping) or not loaded:
        raise ValueError(
            f"{path} does not contain a state dictionary: expected a non-empty mapping, "
            f"found {type(loaded).__name__}"
        )
    if "mot" in loaded:
        return _upstream_tensors(loaded, path)
    if all(torch.is_tensor(value) for value in loaded.values()):
        return dict(loaded)
    offenders = sorted(str(key) for key, value in loaded.items() if not torch.is_tensor(value))
    raise ValueError(
        f"{path} has an unsupported layout: expected tensors for every entry, found {offenders[:10]}. "
        "Supported layouts are a native state dictionary and an upstream PatchWAM release "
        "checkpoint holding 'mot' and 'proprio_encoder' state dictionaries."
    )


def _candidate_names(name):
    """Native names a pretrained weight may map to, most specific first."""
    names = [name]
    names.append(
        name[len(POLICY_KEY_PREFIX):] if name.startswith(POLICY_KEY_PREFIX) else POLICY_KEY_PREFIX + name
    )
    for candidate in tuple(names):
        for leaf in ("weight", "bias"):
            if candidate.endswith("." + leaf):
                # DiT LoRA keeps each pretrained projection as the frozen ``base``.
                names.append(f"{candidate[: -len(leaf)]}base.{leaf}")
    return names


def _target_index(model):
    """Checkpoint keys of a model, with their policy-state or parameter target."""
    policy, extra = _policy_layout(model)
    index = {}
    for key in policy_weight_state(model):
        if not extra:
            index[key] = ("policy", key)
        elif key.startswith(POLICY_KEY_PREFIX):
            index[key] = ("policy", key[len(POLICY_KEY_PREFIX):])
        else:
            index[key] = ("parameter", key)
    return policy, extra, index


def _compatible_weights(model, weights):
    """Validate every checkpoint tensor against the model before any mutation."""
    policy, _, index = _target_index(model)
    expected = policy_weight_state(model)
    mapped, missing, unexpected = {}, [], []
    for name, tensor in weights.items():
        target = next((candidate for candidate in _candidate_names(name) if candidate in index), None)
        if target is None:
            unexpected.append(name)
            continue
        if target in mapped:
            raise ValueError(f"Checkpoint contains conflicting weights for {target}: {name}")
        if tensor.shape != expected[target].shape:
            raise ValueError(
                f"Weight shape differs for {target}: {tuple(tensor.shape)} vs {tuple(expected[target].shape)}"
            )
        if expected[target].is_floating_point() and (
            not tensor.is_floating_point()
            or not torch.isfinite(tensor.to(dtype=expected[target].dtype)).all()
        ):
            raise ValueError(f"Weight values are invalid for {target}")
        mapped[target] = tensor.detach()
    for target in index:
        if target not in mapped and not adapter_parameter_name(target.removeprefix(POLICY_KEY_PREFIX)):
            missing.append(target)
    if missing or unexpected:
        raise ValueError(f"Incomplete policy weights: missing={sorted(missing)[:10]}, unexpected={sorted(unexpected)[:10]}")
    return mapped


def load_policy_weights(model, path):
    """Load native weights without changing optimizer state.

    Native ``.safetensors`` files must cover the complete policy state exactly.
    Tensor-only ``.pt``/``.pth`` state dictionaries are read with
    ``weights_only=True`` and accept either the native schema or the upstream
    PatchWAM release layout. DiT LoRA adapters are zero initialized and stay
    fresh, so a base checkpoint that predates them still loads completely.
    """
    path = Path(path)
    if path.suffix not in (NATIVE_SUFFIX,) + TORCH_SUFFIXES:
        raise ValueError(
            "Policy weights must use native .safetensors or a tensor-only .pt/.pth state dictionary"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    weights = _read_policy_weights(path)
    policy, extra, index = _target_index(model)
    mapped = _compatible_weights(model, weights)
    policy_weights = {
        target: mapped[key] for key, (kind, target) in index.items() if kind == "policy" and key in mapped
    }
    incompatible = policy.load_state_dict(policy_weights, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Checkpoint does not match the policy: unexpected={incompatible.unexpected_keys}")
    refreshed = [name for name in incompatible.missing_keys if not adapter_parameter_name(name)]
    if refreshed:
        raise ValueError(f"Incomplete policy weights: missing={sorted(refreshed)[:10]}")
    with torch.no_grad():
        for name, parameter in extra.items():
            parameter.copy_(mapped[name])
    return {"loaded_tensors": len(weights)}


def register_compact_policy_state(accelerator):
    """Save trainable weights; reload frozen assets from config."""

    def compact(model):
        model = accelerator.unwrap_model(model)
        return (
            model
            if all(hasattr(model, key) for key in ("policy", "autoencoder", "text_encoder"))
            else None
        )

    def save_hook(models, weights, output_dir):
        for index in range(len(models) - 1, -1, -1):
            model = compact(models[index])
            if model is not None:
                if accelerator.is_main_process:
                    state = {
                        key: tensor.detach().cpu().clone().contiguous()
                        for key, tensor in policy_weight_state(model).items()
                    }
                    save_file(state, str(Path(output_dir) / f"policy_{index}.safetensors"))
                weights.pop(index)

    def load_hook(models, input_dir):
        for index in range(len(models) - 1, -1, -1):
            model = compact(models[index])
            if model is not None:
                load_policy_weights(model, Path(input_dir) / f"policy_{index}.safetensors")
                models.pop(index)

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)


def save_policy_average(average, model, path):
    """Export EMA weights separately from training state."""
    save_file(
        average.native_weights(model),
        str(path),
        metadata={
            "weight_variant": "ema",
            "successful_updates": str(average.updates),
        },
    )
