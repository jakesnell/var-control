import argparse
from argparse import Namespace
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Tuple, Optional
import math
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import TensorDataset
import pandas as pd

from var_control.coco_features import load_features_and_ground_truth
from var_control.methods import Rcps, Ltt, VarControl
from var_control.metric import sensitivity, balanced_accuracy
from var_control.split import even_split_iterator


def quantile_indices(x: Tensor, beta: float) -> Tensor:
    assert x.dim() == 1
    n = x.size(0)
    n_trunc = math.floor(n * beta)
    _, inds = torch.sort(x)
    return inds[:n_trunc]


class Batch:
    def __init__(self, args: list[Tensor], num_hypotheses: int):
        assert len(args) == 2
        pred_size, loss = args
        assert pred_size.dim() == 2
        assert pred_size.size(0) == loss.size(0) == num_hypotheses
        assert pred_size.size(1) == loss.size(1)
        self.pred_size = pred_size
        self.loss = loss


def predict(thresholds: Tensor, scores: Tensor) -> Tensor:
    """Use the provided thresholds to create prediction sets from the given
    scores.

    Args:
        thresholds (Tensor) [H] (float): Input thresholds.
        scores (Tensor) [...] (float): Scores to be thresholded.

    Returns:
        Tensor [H, ...] (long): Binary mask of predictions, constructed by thresholding
        the scores.
    """
    preds = torch.gt(scores.unsqueeze(-1), thresholds)
    return torch.permute(preds, [-1] + list(range(scores.dim()))).long()


@dataclass(frozen=True)
class MonotonicTrialResult:
    quantile_loss: float
    mean_in_quantile_loss: float
    mean_in_quantile_pred_size: float
    quantile_satisfied: bool
    mean_loss: float
    mean_pred_size: float
    mean_satisfied: bool


def run_monotonic_trial(
    train_batch: Batch,
    test_batch: Batch,
    method_dict: OrderedDict,
    alpha: float,
    beta: float,
    delta: float,
    markov_scaling: bool,
) -> OrderedDict[str, MonotonicTrialResult]:
    def get_result(method):
        if hasattr(method, "fit_target_var"):
            hypothesis_ind = method.fit_target_var(train_batch.loss, delta, beta, alpha)
        else:
            # use Markov scaling
            if markov_scaling:
                scaled_alpha = alpha * (1 - beta)
            else:
                scaled_alpha = alpha
            hypothesis_ind = method.fit_target_risk(
                train_batch.loss, delta, scaled_alpha
            )

        # compute the test loss
        test_loss = test_batch.loss[hypothesis_ind]
        test_pred_size = test_batch.pred_size[hypothesis_ind]

        mean_loss = test_loss.mean(-1).item()
        mean_pred_size = test_pred_size.float().mean(-1).item()
        mean_satisfied = mean_loss <= alpha

        quantile_loss = torch.quantile(test_loss, beta).item()
        quantile_satisfied = quantile_loss <= alpha

        inds = quantile_indices(test_loss, beta)
        mean_in_quantile_loss = test_loss[inds].mean(-1).item()
        mean_in_quantile_pred_size = test_pred_size[inds].float().mean(-1).item()

        return MonotonicTrialResult(
            mean_loss=mean_loss,
            mean_pred_size=mean_pred_size,
            mean_satisfied=mean_satisfied,
            quantile_loss=quantile_loss,
            mean_in_quantile_loss=mean_in_quantile_loss,
            mean_in_quantile_pred_size=mean_in_quantile_pred_size,
            quantile_satisfied=quantile_satisfied,
        )

    return OrderedDict([(k, get_result(v)) for k, v in method_dict.items()])


@dataclass(frozen=True)
class GeneralTrialResult:
    quantile_loss: float
    mean_in_quantile_loss: float
    mean_in_quantile_pred_size: float
    quantile_satisfied: bool
    mean_loss: float
    mean_pred_size: float
    mean_satisfied: bool
    mean_alpha: float
    quantile_alpha: float


def run_general_trial(
    train_batch: Batch,
    test_batch: Batch,
    method_dict: OrderedDict,
    alpha: float,
    beta: float,
    delta: float,
    markov_scaling: bool,
) -> OrderedDict[str, GeneralTrialResult]:
    def get_result(method):
        if hasattr(method, "fit_var"):
            hypothesis_ind, quantile_alpha = method.fit_var(
                train_batch.loss, delta, beta
            )
            mean_alpha = math.nan
        else:
            if markov_scaling:
                hypothesis_ind = method.fit_risk(
                    train_batch.loss, alpha * (1 - beta), delta
                )
                mean_alpha = math.nan
                quantile_alpha = alpha
            else:
                hypothesis_ind = method.fit_risk(train_batch.loss, alpha, delta)
                mean_alpha = alpha
                quantile_alpha = math.nan

        # compute the test loss
        test_loss = test_batch.loss[hypothesis_ind]
        test_pred_size = test_batch.pred_size[hypothesis_ind]

        mean_loss = test_loss.mean(-1).item()
        mean_pred_size = test_pred_size.float().mean(-1).item()
        mean_satisfied = mean_loss <= mean_alpha

        quantile_loss = torch.quantile(test_loss, beta).item()
        quantile_satisfied = quantile_loss <= quantile_alpha

        inds = quantile_indices(test_loss, beta)
        mean_in_quantile_loss = test_loss[inds].mean(-1).item()
        mean_in_quantile_pred_size = test_pred_size[inds].float().mean(-1).item()

        return GeneralTrialResult(
            mean_loss=mean_loss,
            mean_pred_size=mean_pred_size,
            mean_satisfied=mean_satisfied,
            quantile_loss=quantile_loss,
            mean_in_quantile_loss=mean_in_quantile_loss,
            mean_in_quantile_pred_size=mean_in_quantile_pred_size,
            quantile_satisfied=quantile_satisfied,
            mean_alpha=mean_alpha,
            quantile_alpha=quantile_alpha,
        )

    return OrderedDict([(k, get_result(v)) for k, v in method_dict.items()])


def main(args: Namespace):
    torch.manual_seed(args.seed)

    if args.loss == "sensitivity":
        loss_fn = lambda pred, y: 1 - sensitivity(pred, y)
        loss_type = "monotonic"
        methods = [Rcps, Ltt, VarControl]
    elif args.loss == "balanced_accuracy":
        loss_fn = lambda pred, y: 1 - balanced_accuracy(pred, y)
        loss_type = "general"
        methods = [Ltt, VarControl]
    else:
        raise ValueError(f"Unexpected loss {args.loss}")

    # y: [N, C]
    z, y = load_features_and_ground_truth("tresnet_m")

    thresholds = torch.linspace(z.min().item(), z.max().item(), args.num_hypotheses)

    # make predictions on the entire set
    preds = predict(thresholds, z).permute(1, 0, 2)  # [N, H, C] (long)

    loss = loss_fn(preds, y.unsqueeze(1))

    method_dict = OrderedDict([(method.__name__, method) for method in methods])

    trial_results = []
    for train_sample, test_sample in tqdm(
        even_split_iterator(TensorDataset(preds.sum(-1), loss), args.num_trials),
        total=args.num_trials,
    ):
        train_batch = Batch([x.t() for x in train_sample], args.num_hypotheses)
        test_batch = Batch([x.t() for x in test_sample], args.num_hypotheses)
        if loss_type == "monotonic":
            trial_results.append(
                run_monotonic_trial(
                    train_batch,
                    test_batch,
                    method_dict,
                    alpha=args.alpha,
                    beta=args.beta,
                    delta=args.delta,
                    markov_scaling=args.markov_scaling,
                )
            )
        elif loss_type == "general":
            trial_results.append(
                run_general_trial(
                    train_batch,
                    test_batch,
                    method_dict,
                    alpha=args.alpha,
                    beta=args.beta,
                    delta=args.delta,
                    markov_scaling=args.markov_scaling,
                )
            )
        else:
            raise ValueError(f"Unexpected loss type {loss_type}")

    rows = []
    for trial_ind, trial_result in enumerate(trial_results):
        for k, v in trial_result.items():
            rows.append({"trial": trial_ind + 1, "method": k} | asdict(v))

    results_df = pd.DataFrame(rows)
    results_df.to_csv(f"output/coco_experiments/results_{args.loss}.csv")

    avg_results_df = results_df.drop(columns="trial").groupby(["method"]).mean()
    avg_results_df.to_csv(f"output/coco_experiments/avg_results_{args.loss}.csv")
    print(avg_results_df)
    print(avg_results_df.to_latex(float_format="%.3f"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COCO experiments")
    parser.add_argument(
        "loss", type=str, help="Loss function (options: sensitivity, balanced_accuracy"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--num_hypotheses",
        type=int,
        default=500,
        help="number of hypotheses (default: 500)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1000,
        help="number of random splits (default: 1000)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="target loss value (default: 0.2)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.9, help="target quantile level (default: 0.9)"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="acceptable probability of error (default: 0.05)",
    )
    parser.add_argument(
        "--markov_scaling",
        action="store_true",
        help="use Markov's inequality to convert risk controlling to var controlling",
    )
    args = parser.parse_args()
    main(args)
