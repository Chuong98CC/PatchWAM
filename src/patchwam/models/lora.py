"""Low-rank residual adapters for frozen linear layers.

Vision-language conditioning wraps Qwen projections, and the DiT LoRA mode wraps
the official FLUX.2 block projections. Both paths share this primitive.
"""

import math

from torch import nn

# Adapter tensors keep the ``LowRankResidualLinear`` attribute names below the
# wrapped module path, so a parameter name is enough to recognize an adapter.
ADAPTER_PARAMETER_SUFFIXES = (".residual_in.weight", ".residual_out.weight")

# Official FLUX.2 stacks that hold attention and feed-forward projections.
BLOCK_CONTAINERS = ("double_blocks", "single_blocks")
# Projections expected from the official block contract at the pinned revision.
PROJECTION_LEAVES = ("qkv", "proj", "linear1", "linear2")


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
    """Whether a state-dict or parameter name belongs to a low-rank adapter."""
    return name.endswith(ADAPTER_PARAMETER_SUFFIXES)


def validate_adapter_settings(rank, alpha, dropout, *, prefix: str):
    """Reject unusable low-rank settings with the offending argument named."""
    if type(rank) is not int or rank < 1:
        raise ValueError(f"{prefix}_rank must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"{prefix}_alpha must be finite and positive")
    if not 0 <= dropout < 1:
        raise ValueError(f"{prefix}_dropout must be in [0,1)")


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
    """Wrap every block projection with a zero-initialized low-rank residual.

    Wrapping freezes each projection's base weight only. Parameters that are not
    wrapped keep whatever ``requires_grad`` state they already have.
    """
    validate_adapter_settings(rank, alpha, dropout, prefix="dit_lora")
    names = block_projection_names(transformer)
    for name in names:
        parent_path, _, attribute = name.rpartition(".")
        parent = transformer.get_submodule(parent_path)
        setattr(parent, attribute, LowRankResidualLinear(getattr(parent, attribute), rank, alpha, dropout))
    return names
