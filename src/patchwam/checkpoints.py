"""Policy weights and compact training checkpoints."""

import os
import random
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np
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

RNG_STATE_PREFIX = "random_states_"
RNG_STATE_KEYS = ("step", "random_state", "numpy_random_seed", "torch_manual_seed")


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
    except Exception as error:
        raise _unreadable(path, error) from error
    # Legacy non-zip files cannot be memory mapped; the same safe loader still applies.
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as error:
        raise _unreadable(path, error) from error


# Accelerate writes the process random state (``random``, NumPy, and the torch
# generators) with plain ``torch.save`` and reloads it through ``torch.load``
# without naming a ``weights_only`` mode. Since torch 2.6 that mode defaults to
# ``True``, which rejects the NumPy array inside the saved state; Accelerate
# swallows the failure, silently drops every random state, and then raises
# ``KeyError: 'step'``. Allow exactly the NumPy globals those files need, the
# same way the LIBERO initial state reader does, never by disabling weights-only
# loading.
def _rng_state_globals():
    from numpy.core.multiarray import _reconstruct

    return (
        np.ndarray, _reconstruct, np.dtype,
        type(np.dtype(np.uint32)), type(np.dtype(np.float64)),
    )


@contextmanager
def safe_rng_state_loading():
    """Allow only the NumPy types an Accelerate random state file contains."""
    if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "").lower() in {"1", "y", "yes", "true"}:
        raise ValueError(
            "Resuming requires weights-only loading; remove TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        )
    with torch.serialization.safe_globals(_rng_state_globals()):
        yield


def rng_state_path(directory, process_index):
    """The random state file of one process inside a complete run directory."""
    return Path(directory) / f"{RNG_STATE_PREFIX}{process_index}.pkl"


def _unreadable_rng_state(path, error):
    return ValueError(
        f"Could not read the random state in {path} with torch.load(weights_only=True): "
        f"{type(error).__name__}: {error}. PatchWAM never falls back to unsafe pickle "
        "deserialization; resume from a complete run directory written by this training loop."
    )


def _validate_rng_state(states, path):
    if not isinstance(states, Mapping):
        raise ValueError(
            f"{path} has an unsupported random state layout: expected a mapping, "
            f"found {type(states).__name__}"
        )
    missing = [key for key in RNG_STATE_KEYS if key not in states]
    if missing:
        raise ValueError(f"{path} has an unsupported random state layout: missing {sorted(missing)}")
    devices = states.get("torch_cuda_manual_seed")
    if (
        type(states["step"]) is not int
        or states["step"] < 0
        or not isinstance(states["random_state"], tuple)
        or not isinstance(states["numpy_random_seed"], tuple)
        or len(states["numpy_random_seed"]) != 5
        or not isinstance(states["numpy_random_seed"][1], np.ndarray)
        or not torch.is_tensor(states["torch_manual_seed"])
        or (
            devices is not None
            and (
                not isinstance(devices, (list, tuple))
                or not all(torch.is_tensor(one) for one in devices)
            )
        )
    ):
        raise ValueError(f"{path} has an unsupported random state layout")
    return states


def read_rng_state(directory, process_index):
    """Read one process's random state with weights-only deserialization."""
    path = rng_state_path(directory, process_index)
    if not path.is_file():
        raise ValueError(
            f"{path} is missing: exact continuation needs the saved random state of every "
            "process, so resume from a complete run directory written by this training loop"
        )
    with safe_rng_state_loading():
        try:
            states = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as error:  # noqa: BLE001 - every unreadable or unsafe file is reported alike
            raise _unreadable_rng_state(path, error) from error
    return _validate_rng_state(states, path)


def _rng_state_mismatches(states):
    mismatches = []
    if states["random_state"] != random.getstate():
        mismatches.append("the Python random module")
    saved_numpy = states["numpy_random_seed"]
    live_numpy = np.random.get_state()
    if (
        saved_numpy[0] != live_numpy[0]
        or not np.array_equal(saved_numpy[1], live_numpy[1])
        or tuple(saved_numpy[2:]) != tuple(live_numpy[2:])
    ):
        mismatches.append("NumPy")
    if not torch.equal(states["torch_manual_seed"], torch.get_rng_state()):
        mismatches.append("the torch CPU generator")
    saved_devices = states.get("torch_cuda_manual_seed")
    if saved_devices and torch.cuda.is_available():
        live_devices = torch.cuda.get_rng_state_all()
        if len(saved_devices) != len(live_devices) or not all(
            torch.equal(saved, live) for saved, live in zip(saved_devices, live_devices)
        ):
            mismatches.append("the torch CUDA generators")
    return mismatches


def require_restored_rng_state(states):
    """Fail loudly unless the live process random state is the saved one.

    ``Accelerator.load_state`` swallows every failure while restoring random
    state, so a continuation could otherwise restart on a fresh random stream.
    """
    mismatches = _rng_state_mismatches(states)
    if mismatches:
        raise ValueError(
            "The resumed run did not restore the saved "
            + ", ".join(mismatches)
            + "; continuing would not reproduce the interrupted run"
        )


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
    mot = loaded["mot"]
    values = loaded.get(UPSTREAM_STATE_PROJECTION)
    if not isinstance(mot, Mapping) or (values is not None and not isinstance(values, Mapping)):
        raise ValueError(
            f"{path} has an unsupported layout: expected state dictionaries for 'mot' and "
            f"'{UPSTREAM_STATE_PROJECTION}', found {type(mot).__name__} and {type(values).__name__}"
        )
    if values is None:
        raise ValueError(
            f"{path} uses an unsupported upstream layout: 'mot' requires a "
            f"'{UPSTREAM_STATE_PROJECTION}' state dictionary containing 'weight' and 'bias'"
        )
    step = loaded.get("step")
    if step is not None and (not isinstance(step, int) or isinstance(step, bool) or step < 0):
        raise ValueError(f"{path} has an unsupported layout: 'step' must be a nonnegative integer")
    dtype = loaded.get("torch_dtype")
    if dtype is not None and not isinstance(dtype, str):
        raise ValueError(f"{path} has an unsupported layout: 'torch_dtype' must be a string")
    tensors, aliases, unexpected = {}, {}, []
    for key, value in mot.items():
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


# Legacy custom DiT LoRA tensors and their PEFT LORA destinations. The custom
# adapter held a low-rank input and output projection; PEFT names the same pair
# ``lora_A``/``lora_B`` of the emitted module behind a ``base_layer``.
LEGACY_LORA_DESTINATIONS = {
    "residual_in": "lora_A.default.weight",
    "residual_out": "lora_B.default.weight",
}
LEGACY_ADAPTER_LEAVES = tuple(f".{leaf}.weight" for leaf in LEGACY_LORA_DESTINATIONS)


def _candidate_names(name):
    """Native names a pretrained weight may map to, most specific first."""
    names = [name]
    names.append(
        name[len(POLICY_KEY_PREFIX):] if name.startswith(POLICY_KEY_PREFIX) else POLICY_KEY_PREFIX + name
    )
    for candidate in tuple(names):
        if ".base." in candidate:
            # PEFT keeps the pretrained projection behind ``base_layer``.
            names.append(candidate.replace(".base.", ".base_layer."))
        for legacy, destination in LEGACY_LORA_DESTINATIONS.items():
            suffix = f".{legacy}.weight"
            if candidate.endswith(suffix):
                names.append(f"{candidate[: -len(suffix)]}.{destination}")
        for leaf in ("weight", "bias"):
            if candidate.endswith("." + leaf):
                # PEFT keeps each pretrained projection as the frozen ``base_layer``;
                # the local ``LowRankResidualLinear`` names it ``base``.
                names.append(f"{candidate[: -len(leaf)]}base_layer.{leaf}")
                names.append(f"{candidate[: -len(leaf)]}base.{leaf}")
    return list(dict.fromkeys(names))


def _legacy_adapter_names(weights):
    return sorted(name for name in weights if name.endswith(LEGACY_ADAPTER_LEAVES))


def _target_index(model):
    """Checkpoint keys of a model, with their policy-state or parameter target."""
    _, extra = _policy_layout(model)
    index = {}
    for key in policy_weight_state(model):
        if not extra:
            index[key] = ("policy", key)
        elif key.startswith(POLICY_KEY_PREFIX):
            index[key] = ("policy", key[len(POLICY_KEY_PREFIX):])
        else:
            index[key] = ("parameter", key)
    return index


def _compatible_weights(model, weights):
    """Validate every checkpoint tensor against the model before any mutation."""
    index = _target_index(model)
    expected = policy_weight_state(model)
    unmapped_adapters = [
        name for name in _legacy_adapter_names(weights)
        if not any(candidate in index for candidate in _candidate_names(name))
    ]
    if unmapped_adapters:
        raise ValueError(
            f"{unmapped_adapters[:5]} are custom DiT LoRA adapter tensors that cannot be translated "
            "to a PEFT LORA adapter of this model. Configure a matching PEFT LORA adapter (dit_lora "
            "or dit_adapter_config with peft_type LORA) whose rank matches the checkpoint, export the "
            "weights without adapter tensors, or keep using the checkpoint they came from. Adapter "
            "tensors are never dropped silently."
        )
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
    ``weights_only=True`` and accept the native schema, the upstream PatchWAM
    release layout, or a checkpoint written by the earlier custom DiT LoRA
    implementation. Pretrained projections land on the PEFT ``base_layer`` and
    fresh adapters keep their initialization, so a base checkpoint that predates
    them still loads completely. Legacy custom adapter tensors are translated
    only when the model carries a PEFT LORA adapter of the same rank at that
    projection; tensor-only checkpoints do not record ``lora_alpha``, so the
    configured alpha of the model being loaded remains in force.
    """
    path = Path(path)
    if path.suffix not in (NATIVE_SUFFIX,) + TORCH_SUFFIXES:
        raise ValueError(
            "Policy weights must use native .safetensors or a tensor-only .pt/.pth state dictionary"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    weights = _read_policy_weights(path)
    policy, extra = _policy_layout(model)
    index = _target_index(model)
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
