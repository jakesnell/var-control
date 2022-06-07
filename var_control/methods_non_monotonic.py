from var_control.raps import *
from var_control.raps_utils import *
from var_control.utils import *
from typing import Callable, TypeAlias, Tuple, Optional
import matplotlib.pyplot as plt
from torch import Tensor
from dataclasses import asdict, dataclass
import math
from torch.utils.data import TensorDataset, DataLoader


@dataclass(frozen=True)
class TrialResult:
    mean_loss: float
    quantile_loss: float
    max_loss: float
    mean_prediction_set_size: float
    alpha: float
    mean_in_quantile: float
    avg_quantile_set_size: float
    satisfied_m_target: float
    satisfied_q_target: float


@dataclass(frozen=True)
class Ltt:
    @staticmethod
    def fit_target_var(
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
        thresholds: list,
    ) -> Tuple[Tensor, float]:
        # assumption is that loss increases across the rows, so choose the largest
        # hypothesis index that satisfies the quantile constraint
        num_hypotheses = loss.size(0)
        num_train = loss.size(-1)
        inflated_beta = beta + math.sqrt(
            math.log(num_hypotheses / delta) / (2 * num_train)
        )
        inflated_quantile = torch.quantile(loss, inflated_beta, dim=-1)

        hypothesis_ind = torch.argmin(inflated_quantile)
        alpha = torch.min(inflated_quantile).item()

        return hypothesis_ind, alpha


@dataclass(frozen=True)
class MinQuantile:
    @staticmethod
    def fit_target_var(loss: Tensor, beta: float) -> int:
        emp_quantile = torch.quantile(loss, beta, dim=-1)
        hypothesis_ind = torch.argmin(emp_quantile)

        return hypothesis_ind


@dataclass(frozen=True)
class MinLoss:
    @staticmethod
    def fit_target_var(loss: Tensor) -> Tensor:
        mean_loss = torch.mean(loss, dim=-1)
        hypothesis_ind = torch.argmin(mean_loss)

        return hypothesis_ind


@dataclass(frozen=True)
class Raps:
    @staticmethod
    def fit_target_var(
        z_train: Tensor, y_train: Tensor, z_test: Tensor, target: float
    ) -> Tensor:

        n_classes = y_train.shape[1]

        y_train_labels = torch.argmax(y_train, dim=1)

        calib_dataset = TensorDataset(z_train, y_train_labels)
        calib_loader = DataLoader(
            calib_dataset, batch_size=16, shuffle=True, pin_memory=True
        )

        calib_model = ConformalModelLogits(
            None, calib_loader, alpha=target, lamda_criterion="size"
        )

        _, s = calib_model(z_test)

        predictions = torch.zeros((len(s), n_classes), dtype=int)
        for i, prediction_list in enumerate(s):
            predictions[i, prediction_list.tolist()] = 1

        return predictions


@dataclass(frozen=True)
class Rcps:
    @staticmethod
    def fit_target_var(
        loss: Tensor,
        delta: float,
        beta: float,
        target: float,
    ) -> Tensor:
        num_train = loss.size(-1)
        mean_loss = torch.mean(loss, dim=-1)
        # compute upper confidence bound on risk for each hypothesis
        ucb = mean_loss + math.sqrt(math.log(1.0 / delta) / (2 * num_train))
        # select smallest threshold with ucb less than target
        hypothesis_ind = int(torch.nonzero(ucb <= target).view(-1).max().item())

        return hypothesis_ind


@dataclass(frozen=True)
class ConformalPred:
    @staticmethod
    def fit_target_var(
        z_train: Tensor, y_train: Tensor, z_test: Tensor, target: float
    ) -> Tensor:

        n = z_train.shape[0]
        # collapse one hot label matrix to label vector
        y_train_labels = torch.argmax(y_train, dim=1).tolist()
        # get loss on the ground truth probability for each sample
        scores = 1 - z_train.softmax(dim=1)
        scores = scores[torch.arange(z_train.shape[0]), y_train_labels]
        # compute qhat
        qhat = torch.quantile(scores, np.ceil((n + 1) * (1 - target)) / n)
        # take all predictions with probability greater than (1-qhat)
        predictions = (z_test.softmax(dim=1) > (1 - qhat)).int()

        return predictions


@dataclass(frozen=True)
class Aps:
    @staticmethod
    def fit_target_var(
        z_train: Tensor, y_train: Tensor, z_test: Tensor, target: float
    ) -> Tensor:

        n, n_classes = z_train.shape
        # collapse one hot label matrix to label vector
        y_train_labels = torch.argmax(y_train, dim=1).tolist()
        # sort val probabilities
        sorted, pi = z_train.softmax(dim=1).sort(dim=1, descending=True)
        # get a vector of cumulative probabilities needed to include ground truth
        scores = sorted.cumsum(dim=1).gather(1, pi.argsort(1))[
            range(y_train.shape[0]), y_train_labels
        ]
        # compute qhat
        qhat = torch.quantile(scores, np.ceil((n + 1) * (1 - target)) / n)

        # sort test probabilities
        test_sorted, test_pi = z_test.softmax(dim=1).sort(dim=1, descending=True)
        # get number of predictions per sample needed to get cumsum above qhat
        sizes = (test_sorted.cumsum(dim=1) > qhat).int().argmax(dim=1)
        # make list of predictions per sample, convert to matrix (this could be coded better)
        prediction_lists = [test_pi[i][: (sizes[i] + 1)] for i in range(sizes.shape[0])]
        predictions = torch.zeros((len(prediction_lists), n_classes), dtype=int)
        for i, prediction_list in enumerate(prediction_lists):
            predictions[i, prediction_list] = 1

        return predictions
