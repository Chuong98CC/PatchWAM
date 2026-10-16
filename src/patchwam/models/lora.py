"""Projection adapters for the frozen vision-language and DiT backbones.

Vision-language conditioning wraps Qwen projections in the local
``LowRankResidualLinear`` primitive. DiT finetuning uses Hugging Face PEFT, so the
available adapter families follow the PEFT catalog (currently LORA and IA3).

The low-level ``inject_adapter_in_model`` entry point is used on purpose: the
masked manual execution in ``OfficialFluxDenoiser`` calls the official module
attributes (``qkv``, ``proj``, ``linear1``, ``linear2``, ...) directly, and
low-level injection keeps those attributes and their pretrained tensors in place.
See https://huggingface.co/docs/peft/developer_guides/low_level_api for the
documented difference from ``get_peft_model``: no ``PeftModel`` wrapper is
retained around the transformer, so checkpoints stay the policy state dictionary
produced by ``policy_weight_state`` instead of a second, nested wrapper.

PEFT marks every parameter that does not carry the tuner prefix as frozen during
injection, which would silently stop training the rest of the pretrained DiT.
Every injection here restores the pre-injection ``requires_grad`` state of each
original parameter and freezes only the adapted projection weight.
"""

import enum
import importlib
import math
from collections.abc import Mapping, Sequence

from torch import nn

# Adapter tensors of the local ``LowRankResidualLinear`` primitive keep their
# attribute names below the wrapped module path.
ADAPTER_PARAMETER_SUFFIXES = (".residual_in.weight", ".residual_out.weight")

# Attribute markers PEFT adds below an adapted module path. Both families are
# recognized so one predicate covers VL adapters, PEFT LoRA, and PEFT IA3.
PEFT_ADAPTER_MARKERS = (
    ".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B.",
    ".lora_magnitude_vector", ".ia3_l",
)

# Official FLUX.2 stacks that hold attention and feed-forward projections.
BLOCK_CONTAINERS = ("double_blocks", "single_blocks")
# Projections expected from the official block contract at the pinned revision.
PROJECTION_LEAVES = ("qkv", "proj", "linear1", "linear2")

# Adapter families that work with the masked manual execution path.
SUPPORTED_ADAPTER_TYPES = ("LORA", "IA3")
DEFAULT_ADAPTER_TYPE = "LORA"

# Legacy default DiT LoRA settings. An explicit ``dit_adapter_config`` mapping owns
# the fields it declares; these constants let injection reject a mapping that
# contradicts an explicitly requested legacy flag instead of ignoring one of them.
DEFAULT_LORA_RANK = 64
DEFAULT_LORA_ALPHA = 128
DEFAULT_LORA_DROPOUT = 0.05

# Feed-forward projections of the official block contract, which IA3 rescales on
# the input side as in the paper. Attention projections stay on the output side.
FEEDFORWARD_LEAVES = ("linear1", "linear2")
FEEDFORWARD_CONTAINERS = ("img_mlp", "txt_mlp")

_PEFT_MODULE = None


class LowRankResidualLinear(nn.Module):
    """Frozen linear map with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 64, alpha: float = 128, dropout: float = 0.05):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid low-rank adapter settings")
        self.base = base.requires_grad_(False)
        self.residual_in = nn.Linear(base.in_features, rank, bias=False).to(base.weight)
        self.residual_out = nn.Linear(rank, base.out_features, bias=False).to(base.weight)
        nn.init.kaiming_uniform_(self.residual_in.weight, a=math.sqrt(5))
        nn.init.zeros_(self.residual_out.weight)
        self.dropout = nn.Dropout(dropout)
        self.multiplier = alpha / rank

    def forward(self, value):
        return self.base(value) + self.residual_out(self.residual_in(self.dropout(value))) * self.multiplier


def adapter_parameter_name(name: str) -> bool:
    """Whether a state-dict or parameter name belongs to an adapter tensor."""
    if name.endswith(ADAPTER_PARAMETER_SUFFIXES):
        return True
    return any(marker in name for marker in PEFT_ADAPTER_MARKERS)


def validate_adapter_settings(rank, alpha, dropout, *, prefix: str):
    """Reject unusable low-rank settings with the offending argument named."""
    if type(rank) is not int or rank < 1:
        raise ValueError(f"{prefix}_rank must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"{prefix}_alpha must be finite and positive")
    if not 0 <= dropout < 1:
        raise ValueError(f"{prefix}_dropout must be in [0,1)")


def peft_module():
    """Import PEFT on first use so full-mode runs need no adapter dependency."""
    global _PEFT_MODULE
    if _PEFT_MODULE is None:
        try:
            _PEFT_MODULE = importlib.import_module("peft")
        except ImportError as exc:  # pragma: no cover - exercised without the extra installed
            raise ImportError(
                "DiT adapter finetuning requires the optional 'peft' dependency. Install it with "
                "`pip install 'patchwam[flux]'` or `uv sync --extra flux`."
            ) from exc
    return _PEFT_MODULE


def peft_version() -> str:
    """Version of the PEFT package backing the adapter implementation."""
    return str(getattr(peft_module(), "__version__", "unknown"))


def _tuner_layer_types():
    return (peft_module().tuners.tuners_utils.BaseTunerLayer,)


def is_adapter_layer(module: nn.Module) -> bool:
    """Whether a module is an injected PEFT adapter layer."""
    return isinstance(module, _tuner_layer_types())


def linear_projection_names(model: nn.Module) -> list[str]:
    """Every individual ``nn.Linear`` leaf of the official transformer."""
    return [name for name, module in model.named_modules() if name and isinstance(module, nn.Linear)]


def _target_specs(value, field: str) -> list:
    """Normalize a configured target set, including Hydra list-like containers."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (set, frozenset)):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise TypeError(
        f"dit_adapter_config {field} must be a string or a list of module names, "
        f"found {type(value).__name__}"
    )


def resolve_adapter_targets(
    model: nn.Module, target_modules, *, field: str = "target_modules"
) -> list[str]:
    """Resolve configured targets to exact ``nn.Linear`` leaves.

    Entries are exact dotted module paths or dotted-path suffixes such as
    ``img_attn.qkv``. Container modules, and names that do not resolve to an
    individual linear projection, are rejected with the available names listed.
    """
    specs = _target_specs(target_modules, field)
    linears = linear_projection_names(model)
    linear_set = set(linears)
    containers = {
        name for name, module in model.named_modules() if name and not isinstance(module, nn.Linear)
    }
    resolved: set[str] = set()
    for spec in specs:
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError(
                f"Every dit_adapter_config {field} entry must be a non-empty module name, "
                f"found {spec!r}"
            )
        spec = spec.strip()
        if spec in linear_set:
            resolved.add(spec)
            continue
        matches = sorted(name for name in linears if name.endswith("." + spec))
        if matches:
            resolved.update(matches)
            continue
        if spec in containers:
            example = next(
                (name for name in sorted(linears) if name.startswith(spec + ".")),
                f"{spec}.<leaf>",
            )
            raise ValueError(
                f"dit_adapter_config target '{spec}' is a module container, not an individual "
                f"nn.Linear; name the projection itself, for example '{example}'"
            )
        examples = ", ".join(sorted(linears)[:5])
        raise ValueError(
            f"dit_adapter_config target '{spec}' does not match an nn.Linear module of the official "
            f"transformer. Existing projections include: {examples}"
        )
    return sorted(resolved)


def _normalize_adapter_type(value) -> str:
    if isinstance(value, enum.Enum):
        value = value.value
    name = str(value).strip().upper().rsplit(".", 1)[-1]
    if name not in SUPPORTED_ADAPTER_TYPES:
        prompt_types = sorted(
            str(member.value)
            for member in peft_module().PeftType
            if "PROMPT" in member.name or "P_TUNING" in member.name
        )
        extra = (
            f" Prompt and prompt-learning types ({', '.join(prompt_types)}) are not supported."
            if prompt_types
            else ""
        )
        raise ValueError(
            f"Unsupported dit_adapter_config peft_type {value!r}; supported types are "
            f"{', '.join(SUPPORTED_ADAPTER_TYPES)}.{extra}"
        )
    return name


def _reject_unsupported_options(config, adapter_type: str) -> None:
    """Reject PEFT options this execution path cannot keep honest."""
    if config.is_prompt_learning or config.is_adaption_prompt:
        raise ValueError(
            f"dit_adapter_config peft_type {adapter_type!r} is a prompt-learning method; only "
            f"{', '.join(SUPPORTED_ADAPTER_TYPES)} adapters are supported."
        )
    rejected = {
        "modules_to_save": "modules_to_save registers duplicate trainable copies of existing layers",
        "layers_to_transform": "layers_to_transform is not supported; name the projections directly",
        "layers_pattern": "layers_pattern is not supported; name the projections directly",
        "target_parameters": "target_parameters wraps individual tensors instead of linear projections",
        "exclude_modules": "exclude_modules is not supported; list the projections to adapt instead",
    }
    for field, reason in rejected.items():
        if getattr(config, field, None) not in (None, [], (), set(), frozenset(), {}):
            raise ValueError(f"dit_adapter_config {field} is not supported: {reason}")
    if getattr(config, "inference_mode", False):
        raise ValueError("dit_adapter_config inference_mode would freeze the adapters; leave it False")
    if getattr(config, "bias", "none") != "none":
        raise ValueError(
            "dit_adapter_config bias must be 'none': only adapted projection weights are frozen, so an "
            "adapter bias would be silently re-frozen by the training hook"
        )
    if getattr(config, "lora_bias", False):
        raise ValueError(
            "dit_adapter_config lora_bias must be False: only adapted projection weights are frozen, so "
            "an adapter bias would be silently re-frozen by the training hook"
        )
    if getattr(config, "use_dora", False):
        raise ValueError(
            "dit_adapter_config use_dora is not validated against the masked manual execution path; use "
            "LORA or IA3"
        )
    if getattr(config, "init_lora_weights", True) not in (True, False):
        raise ValueError(
            "dit_adapter_config init_lora_weights must be True (zero-initialized up-projection) or False"
        )
    if getattr(config, "init_ia3_weights", True) is not True:
        raise ValueError(
            "dit_adapter_config init_ia3_weights must be True so a fresh adapter starts from the "
            "pretrained function"
        )
    if getattr(config, "fan_in_fan_out", False):
        raise ValueError("dit_adapter_config fan_in_fan_out is not supported for the official projections")


def default_feedforward_modules(targets) -> list[str]:
    """Feed-forward projections of the resolved targets.

    The official FLUX.2 blocks keep their feed-forward projections under
    ``img_mlp``/``txt_mlp`` and fuse the single-stream attention and feed-forward
    path into ``linear1``/``linear2``. IA3 rescales feed-forward projections on the
    input side (its ``feedforward_modules`` convention), so every MLP projection
    and both fused projections are treated as feed-forward, while the attention
    projections stay on the output side, which is where IA3 rescales keys and
    values in the paper. Pass ``feedforward_modules`` explicitly to choose the
    other treatment for a fused projection.
    """
    return sorted(
        name
        for name in targets
        if name.rsplit(".", 1)[-1] in FEEDFORWARD_LEAVES
        or any(part in FEEDFORWARD_CONTAINERS for part in name.split("."))
    )


def _validate_conflicting_flags(mapping: Mapping, adapter_type: str, *, rank, alpha, dropout) -> None:
    """Reject a mapping that contradicts an explicitly requested legacy flag."""
    if adapter_type != "LORA":
        for field, value, default in (
            ("dit_lora_rank", rank, DEFAULT_LORA_RANK),
            ("dit_lora_alpha", alpha, DEFAULT_LORA_ALPHA),
            ("dit_lora_dropout", dropout, DEFAULT_LORA_DROPOUT),
        ):
            if value != default:
                raise ValueError(
                    f"dit_adapter_config peft_type IA3 does not use r/lora_alpha/lora_dropout, so "
                    f"{field}={value!r} cannot be honored; use peft_type LORA or remove that argument"
                )
        return
    for field, key, value, default in (
        ("dit_lora_rank", "r", rank, DEFAULT_LORA_RANK),
        ("dit_lora_alpha", "lora_alpha", alpha, DEFAULT_LORA_ALPHA),
        ("dit_lora_dropout", "lora_dropout", dropout, DEFAULT_LORA_DROPOUT),
    ):
        if key in mapping and value != default and mapping[key] != value:
            raise ValueError(
                f"dit_adapter_config {key}={mapping[key]!r} conflicts with {field}={value!r}; a "
                f"non-default {field} cannot be silently ignored, use one setting or the other"
            )


def validated_adapter_options(
    mapping: Mapping | None, *,
    dit_lora: bool = False, rank: int = DEFAULT_LORA_RANK, alpha: float = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
) -> dict | None:
    """Validate an adapter request and return normalized PEFT options.

    Model-independent, so asset loading can reject an unusable configuration
    before reading weights. ``target_modules`` and ``feedforward_modules`` stay as
    configured here and are resolved against the official transformer later.
    """
    if mapping is None:
        if not dit_lora:
            return None
        validate_adapter_settings(rank, alpha, dropout, prefix="dit_lora")
        return {
            "peft_type": DEFAULT_ADAPTER_TYPE, "r": rank, "lora_alpha": alpha,
            "lora_dropout": dropout,
        }
    if isinstance(mapping, bool) or not isinstance(mapping, Mapping):
        raise TypeError(
            "dit_adapter_config must be a mapping of PEFT options or None, found "
            f"{type(mapping).__name__}"
        )
    if not mapping:
        raise ValueError(
            "dit_adapter_config must not be empty; provide PEFT options such as {'peft_type': 'IA3'} "
            "or omit it entirely"
        )
    options = {str(key): value for key, value in mapping.items()}
    adapter_type = _normalize_adapter_type(options.get("peft_type", DEFAULT_ADAPTER_TYPE))
    options["peft_type"] = adapter_type
    _validate_conflicting_flags(options, adapter_type, rank=rank, alpha=alpha, dropout=dropout)
    for field in ("target_modules", "feedforward_modules"):
        value = options.get(field)
        if value is not None:
            options[field] = _target_specs(value, field)
    if adapter_type == "LORA":
        if {"r", "lora_alpha", "lora_dropout"} & set(options):
            configured_rank = options.get("r", rank)
            configured_alpha = options.get("lora_alpha", alpha)
            configured_dropout = options.get("lora_dropout", dropout)
            if type(configured_rank) is not int or configured_rank < 1:
                raise ValueError("dit_adapter_config r must be a positive integer")
            if not math.isfinite(configured_alpha) or configured_alpha <= 0:
                raise ValueError("dit_adapter_config lora_alpha must be finite and positive")
            if not 0 <= configured_dropout < 1:
                raise ValueError("dit_adapter_config lora_dropout must be in [0,1)")
        else:
            validate_adapter_settings(rank, alpha, dropout, prefix="dit_lora")
    return options


def build_dit_adapter_config(
    model: nn.Module, mapping: Mapping | None, *,
    dit_lora: bool = False, rank: int = DEFAULT_LORA_RANK, alpha: float = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
):
    """Build the effective PEFT config, or ``None`` for the full-finetune mode.

    A non-empty ``dit_adapter_config`` mapping enables PEFT independently of
    ``dit_lora`` and owns every field it declares. Without a mapping, ``dit_lora``
    builds a LORA config from the legacy ``dit_lora_rank``/``dit_lora_alpha``/
    ``dit_lora_dropout`` flags. Without either, no adapter is injected and the
    pretrained transformer keeps its full-training contract.
    """
    options = validated_adapter_options(
        mapping, dit_lora=dit_lora, rank=rank, alpha=alpha, dropout=dropout
    )
    if options is None:
        return None
    adapter_type = options["peft_type"]
    targets = options.get("target_modules")
    if targets is None:
        targets = block_projection_names(model)
        layout = "block_attention_and_mlp_projections"
    else:
        targets = resolve_adapter_targets(model, targets)
        layout = "explicit_target_modules"
    if not targets:
        raise ValueError("dit_adapter_config resolved to no adapter targets")
    options["target_modules"] = targets
    if adapter_type == "IA3":
        feedforward = options.get("feedforward_modules")
        feedforward = (
            default_feedforward_modules(targets) if feedforward is None
            else resolve_adapter_targets(model, feedforward, field="feedforward_modules")
        )
        outside = sorted(set(feedforward) - set(targets))
        if outside:
            raise ValueError(
                "dit_adapter_config feedforward_modules must be a subset of target_modules; "
                f"{outside} is not adapted"
            )
        options["feedforward_modules"] = feedforward
    try:
        config = peft_module().get_peft_config(options)
    except ValueError as exc:
        raise ValueError(f"dit_adapter_config is not a usable PEFT configuration: {exc}") from exc
    _reject_unsupported_options(config, adapter_type)
    config.adapter_layout = layout
    return config


def serialized_adapter_config(config) -> dict:
    """JSON-safe view of an effective PEFT config for contracts and reports."""
    def normalize(value):
        if isinstance(value, enum.Enum):
            return value.value
        if isinstance(value, (set, frozenset)):
            return sorted(str(item) for item in value)
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    return normalize(config.to_dict())


def adapter_layer_paths(model: nn.Module) -> list[str]:
    """Module paths of the injected PEFT adapter layers."""
    return [
        name for name, module in model.named_modules() if name and isinstance(module, _tuner_layer_types())
    ]


def adapter_parameter_layers(model: nn.Module) -> dict[str, str]:
    """Adapter parameter names mapped to the adapter layer that owns them.

    Uses PEFT's own layer metadata (the base-layer parameter set) instead of name
    heuristics, so IA3 parameters are reported as IA3 parameters and not as LoRA
    tensors.
    """
    result: dict[str, str] = {}
    for path, module in model.named_modules():
        if not path or not isinstance(module, _tuner_layer_types()):
            continue
        base = module.get_base_layer()
        base_ids = {id(parameter) for parameter in base.parameters()}
        for name, parameter in module.named_parameters():
            if id(parameter) not in base_ids:
                result[f"{path}.{name}"] = path
    return result


def _restore_training_flags(model: nn.Module, snapshot: dict[int, bool]) -> None:
    """Freeze adapted projection weights and restore every other original flag."""
    frozen_weights: set[int] = set()
    adapter_parameters: set[int] = set()
    for module in model.modules():
        if not isinstance(module, _tuner_layer_types()):
            continue
        base = module.get_base_layer()
        base_ids = {id(parameter) for parameter in base.parameters()}
        for name, parameter in base.named_parameters(recurse=False):
            if name == "weight":
                frozen_weights.add(id(parameter))
        for parameter in module.parameters():
            if id(parameter) not in base_ids:
                adapter_parameters.add(id(parameter))
    for parameter in model.parameters():
        identity = id(parameter)
        if identity in frozen_weights:
            parameter.requires_grad_(False)
        elif identity in adapter_parameters:
            parameter.requires_grad_(True)
        else:
            parameter.requires_grad_(snapshot.get(identity, parameter.requires_grad))


def attach_adapter_layers(model: nn.Module, config) -> list[str]:
    """Inject PEFT adapters and restore the original training flags.

    Returns the exact adapted projection paths. Only the adapted projection weight
    of each target stays frozen; every parameter that was trainable before
    injection is trainable afterwards.
    """
    if config is None:
        raise ValueError("An effective PEFT adapter configuration is required")
    targets = sorted(config.target_modules)
    existing = adapter_layer_paths(model)
    if existing:
        raise ValueError(
            f"The transformer already carries PEFT adapter layers at {existing[:3]}; inject adapters once"
        )
    linears = set(linear_projection_names(model))
    invalid = [name for name in targets if name not in linears]
    if invalid:
        raise ValueError(f"Adapter targets must be individual nn.Linear modules; invalid: {invalid}")
    snapshot = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    peft_module().inject_adapter_in_model(config, model)
    _restore_training_flags(model, snapshot)
    missing = sorted(set(targets) - set(adapter_layer_paths(model)))
    if missing:
        raise ValueError(f"PEFT did not inject adapters at {missing}")
    return targets


def freeze_adapter_bases(model: nn.Module) -> int:
    """Re-assert frozen adapted weights and trainable adapters only."""
    layers = [
        module for name, module in model.named_modules()
        if name and isinstance(module, _tuner_layer_types())
    ]
    if not layers:
        raise ValueError("The model is not configured for PEFT adapters")
    for module in layers:
        base = module.get_base_layer()
        base_ids = {id(parameter) for parameter in base.parameters()}
        for name, parameter in base.named_parameters(recurse=False):
            if name == "weight":
                parameter.requires_grad_(False)
        for parameter in module.parameters():
            if id(parameter) not in base_ids:
                parameter.requires_grad_(True)
    return len(layers)


def block_projection_names(transformer: nn.Module) -> list[str]:
    """Exact module paths of the block attention and MLP projections."""
    names = []
    for container in BLOCK_CONTAINERS:
        blocks = getattr(transformer, container, None)
        if not isinstance(blocks, (nn.ModuleList, nn.Sequential, list, tuple)) or not len(blocks):
            raise ValueError(f"The official transformer has no usable '{container}' stack to adapt")
        for index, block in enumerate(blocks):
            names.extend(
                f"{container}.{index}.{name}"
                for name, module in block.named_modules()
                if isinstance(module, nn.Linear)
            )
    leaves = {name.rsplit(".", 1)[-1] for name in names}
    missing = sorted(set(PROJECTION_LEAVES) - leaves)
    if missing:
        raise ValueError(
            f"The official block contract is missing projections {missing}; found {sorted(leaves)}"
        )
    return names


def attach_block_adapters(transformer: nn.Module, *, rank: int, alpha: float, dropout: float) -> list[str]:
    """Adapt every block projection with PEFT LORA (the legacy DiT LoRA entry point).

    Injection freezes each adapted projection weight only. Parameters that are not
    adapted keep whatever ``requires_grad`` state they already have.
    """
    config = build_dit_adapter_config(
        transformer, None, dit_lora=True, rank=rank, alpha=alpha, dropout=dropout
    )
    return attach_adapter_layers(transformer, config)
