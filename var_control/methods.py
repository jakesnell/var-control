import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor


@dataclass(frozen=True)
class Rcps:
    @staticmethod
    def fit_target_risk(
        loss: Tensor,
        delta: float,
        target: float,
    ) -> int:
        num_train = loss.size(-1)
        mean_loss = torch.mean(loss, dim=-1)
        # compute upper confidence bound on risk for each hypothesis
        ucb = mean_loss + math.sqrt(math.log(1.0 / delta) / (2 * num_train))
        # select smallest threshold with ucb less than target
        hypothesis_ind = int(torch.nonzero(ucb <= target).view(-1).max().item())

        return hypothesis_ind


@dataclass(frozen=True)
class Ltt:
    @staticmethod
    def fit_target_risk(
        loss: Tensor,
        delta: float,
        target: float,
    ) -> int:
        n = loss.shape[1]
        # compute mean risk per threhold
        risk = loss.permute(1, 0).mean(dim=0)
        # compute p-value
        pvals = torch.exp(-2 * n * (torch.relu(target - risk) ** 2))

        hypothesis_ind = int(
            torch.nonzero(pvals < delta / loss.shape[0]).view(-1).max().item()
        )

        return hypothesis_ind

    @staticmethod
    def fit_risk(
        metric: Tensor,
        target: float,
        delta: float,
    ) -> Optional[Tensor]:

        n = metric.shape[1]
        # compute mean risk per threhold
        risk = metric.permute(1, 0).mean(dim=0)
        # compute p-value
        pvals = torch.exp(-2 * n * (torch.relu(target - risk) ** 2))
        # perform multiple testing with Bonferroni correction
        lambda_hats = risk[pvals < delta / risk.shape[0]]
        if lambda_hats.nelement() == 0:
            hypothesis_ind = None
        else:
            hypothesis_ind = torch.argmin(risk)

        return hypothesis_ind


@dataclass(frozen=True)
class VarControl:
    @staticmethod
    def fit_target_var(
        loss: Tensor,
        delta: float,
        beta: float,
        target: float,
    ) -> int:
        # assumption is that loss increases across the rows, so choose the largest
        # hypothesis index that satisfies the quantile constraint
        num_hypotheses = loss.size(0)
        num_train = loss.size(-1)
        inflated_beta = beta + math.sqrt(
            math.log(num_hypotheses / delta) / (2 * num_train)
        )
        inflated_quantile = torch.quantile(loss, inflated_beta, dim=-1)

        # select hypothesis based on quantile
        hypothesis_ind = int(
            torch.nonzero(inflated_quantile <= target).view(-1).max().item()
        )

        return hypothesis_ind

    @staticmethod
    def fit_var(
        loss: Tensor,
        delta: float,
        beta: float,
    ) -> Tuple[int, float]:
        num_hypotheses = loss.size(0)
        num_train = loss.size(-1)
        inflated_beta = beta + math.sqrt(
            math.log(num_hypotheses / delta) / (2 * num_train)
        )
        inflated_quantile = torch.quantile(loss, inflated_beta, dim=-1)

        hypothesis_ind = torch.argmin(inflated_quantile, -1)
        alpha = inflated_quantile[hypothesis_ind]

        return int(hypothesis_ind.item()), float(alpha.item())
