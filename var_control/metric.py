import torch
from torch import Tensor


def sensitivity(input: Tensor, target: Tensor) -> Tensor:
    """Compute the sensitivity (recall).

    Args:
        input (Tensor) [..., C]: Predictions.
        target (Tensor) [..., C]: Targets.

    Returns:
        Tensor [...]: The per-example sensitivity.
    """
    numer = torch.sum(input * target, -1)
    denom = torch.sum(target, -1)
    ratio = numer / denom
    return torch.where(denom.gt(0), ratio, torch.ones_like(ratio))


def specificity(input: Tensor, target: Tensor) -> Tensor:
    return sensitivity(1 - input, 1 - target)


def balanced_accuracy(input: Tensor, target: Tensor) -> Tensor:
    return 0.5 * (sensitivity(input, target) + specificity(input, target))
