from typing import Callable, TypeAlias
from torch import Tensor
import torch
import numpy as np
import random
from dataclasses import asdict, dataclass
import argparse

Hypothesis: TypeAlias = Callable[[Tensor], Tensor]


def hypothesis_of_threshold(threshold: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def hypothesis(x: torch.Tensor) -> torch.Tensor:
        return x.gt(threshold).long()

    return hypothesis


def construct_hypotheses(
        min_thresh: float, max_thresh: float, num_hypotheses: int
) -> tuple[list[Hypothesis], Tensor]:
    thresholds = torch.linspace(min_thresh, max_thresh, num_hypotheses)
    return list(map(hypothesis_of_threshold, thresholds.tolist())), thresholds


def predict_batch(hypotheses: list[Hypothesis], x: Tensor) -> Tensor:
    return torch.cat([h(x).unsqueeze(0) for h in hypotheses], 0)


def seed_all(seed: int = 36):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
