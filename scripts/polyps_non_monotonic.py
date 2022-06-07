from collections import OrderedDict
from dataclasses import asdict, dataclass
import math
from tqdm import tqdm
from typing import Callable, TypeAlias
import torch
from torch import Tensor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl  # type: ignore
from typing import Tuple, Optional
import argparse
from torch.utils.data import TensorDataset, DataLoader
import pickle

from var_control.polyps_features import load_features_and_ground_truth
from var_control.metric import balanced_accuracy, sensitivity, specificity
from var_control.split import fixed_split
from var_control.methods_non_monotonic import *
from var_control.utils import seed_all

import matplotlib.pyplot as plt  # type: ignore
mpl.use("tkagg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta = None
delta = None
target = None
plotted = False


task_save_path = "output/polyps/"
plot_save_path = task_save_path + "plots/miou/"
results_save_path = task_save_path + "results/"


def plot_var_control(
    metric: Tensor,
    delta: float,
    beta: float,
    target: float,
    thresholds: list,
    save_path: str = None,
):
    # assumption is that loss increases across the rows, so choose the largest
    # hypothesis index that satisfies the quantile constraint
    num_hypotheses = metric.size(0)
    num_train = metric.size(-1)
    inflated_beta = beta + math.sqrt(math.log(num_hypotheses / delta) / (2 * num_train))

    emp_quantile = torch.quantile(metric, beta, dim=-1)
    inflated_quantile = torch.quantile(metric, inflated_beta, dim=-1)
    mean_loss = metric.mean(dim=1)

    lam = -thresholds

    plt.plot(lam, emp_quantile, "-", label=f"emp. quantile @ {beta:0.2f}")
    plt.plot(
        lam, inflated_quantile, "--", label=f"emp. quantile @ {inflated_beta:0.2f}"
    )
    plt.plot(lam, mean_loss, label="mean loss")
    plt.plot(lam, target * torch.ones_like(lam), "-.", color="black")
    plt.xlabel("$\\lambda$")
    plt.ylabel("1 - Balanced Accuracy")
    plt.legend(loc="best")
    plt.savefig(plot_save_path + "beta_" + str(beta) + "_dynamic_var_plot.png", dpi=600)
    plt.clf()
    plt.close()


def var_plot_beta_vs_alpha(metric, loss, delta, z_test, y_test, hypotheses):
    global feature_key
    betas = torch.linspace(0.0, 1.0, 100)
    alphas = []
    mean_losses = []
    for test_beta in betas:
        num_hypotheses = metric.size(0)
        num_train = metric.size(-1)
        inflated_beta = test_beta + math.sqrt(
            math.log(num_hypotheses / delta) / (2 * num_train)
        )
        try:
            inflated_quantile = torch.quantile(metric, inflated_beta, dim=-1)
            hypothesis_ind = torch.argmin(inflated_quantile)
            alpha = torch.min(inflated_quantile).item()
            alphas.append(alpha)

            predictions = get_predictions(hypothesis_ind, z_test, hypotheses)
            predictions = predictions.view(predictions.shape[0], -1)
            y_test_flat = y_test.view(y_test.shape[0], -1)

            # compute the test loss
            test_loss = loss(predictions, y_test_flat)
            # compute mean loss
            mean_loss = test_loss.mean().item()
            mean_losses.append(mean_loss)
        except Exception as e:
            continue

    plt.plot(betas[: len(alphas)], alphas, label="beta vs. alpha")
    plt.plot(betas[: len(alphas)], mean_losses, label="mean loss given beta")
    plt.xlabel("Beta")
    plt.ylabel("Alpha")
    # plt.legend(loc="best")
    plt.legend()
    plt.savefig(plot_save_path + "beta_vs_alpha_bal_acc.png", dpi=600)
    plt.clf()
    plt.close()

    plt.plot(alphas, betas[: len(alphas)], label="alpha vs. beta")
    plt.plot(alphas, mean_losses, label="mean loss given alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Beta")
    # plt.legend(loc="best")
    plt.legend()
    plt.savefig(plot_save_path + "alpha_vs_beta_bal_acc.png", dpi=600)
    plt.clf()
    plt.close()


def get_predictions(hypothesis_ind, z_test, hypotheses):
    if not hypothesis_ind:
        predictions = torch.zeros_like(z_test)
    else:
        h = hypotheses[hypothesis_ind]
        predictions = h(z_test)
    return predictions


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=2):
    with torch.no_grad():

        if len(list(pred_mask.size())) <= 2:
            pred_mask = pred_mask.unsqueeze(0)
        mask = mask.view(1, mask.shape[0], -1)

        iou_per_class = []
        for clas in range(1, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = (
                    torch.logical_and(true_class, true_label).sum(axis=2).float()
                )
                union = torch.logical_or(true_class, true_label).sum(axis=2).float()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou.unsqueeze(0))
        miou = torch.concat(iou_per_class, 0)
        miou = torch.mean(miou, axis=0).squeeze(0)

        return miou


def run_trial(
    z: Tensor,
    y: Tensor,
    loss: Callable[[Tensor, Tensor], Tensor],
    method_dict: OrderedDict,
    hypotheses: list[Hypothesis],
    thresholds: list[float],
    last_trial: bool,
) -> OrderedDict[str, TrialResult]:
    (z_train, y_train), (z_test, y_test) = fixed_split(z, y, 1000)

    metric = []
    batch = 8
    ltt_alpha = 0.3

    for i in range(z_train.shape[0] // batch):
        batch_pred = predict_batch(
            hypotheses, z_train[i * batch : (i + 1) * batch].to(device)
        )

        batch_metric = loss(
            batch_pred.view(batch_pred.shape[0], batch_pred.shape[1], -1).to(device),
            y_train[i * batch : (i + 1) * batch].view(batch, -1).to(device),
        )
        metric.append(batch_metric)

    metric = torch.concat(metric, 1)

    def get_result(method):
        # fit the hypothesis according to the criterion
        global target, beta, delta, plotted
        alpha = None

        if method in [VarControl]:
            hypothesis_ind, alpha = method.fit_target_var(
                metric, delta, beta, thresholds
            )
            target = alpha
            predictions = get_predictions(hypothesis_ind, z_test, hypotheses)
            if last_trial and not plotted:
                var_plot_beta_vs_alpha(
                    metric.cpu(), loss, delta, z_test, y_test, hypotheses
                )
        elif method in [MinLoss]:
            hypothesis_ind = method.fit_target_var(metric)
            predictions = get_predictions(hypothesis_ind, z_test, hypotheses)
        elif method in [Ltt]:
            alpha = ltt_alpha
            hypothesis_ind = method.fit_target_var(metric, alpha, delta)
            predictions = get_predictions(hypothesis_ind, z_test, hypotheses)

        if last_trial and not plotted:
            plot_var_control(
                metric.cpu(), delta, beta, target, thresholds, plot_save_path
            )
            var_plot_beta_vs_alpha(metric, loss, delta, z_test, y_test, hypotheses)
            plotted = True

        predictions = predictions.view(predictions.shape[0], -1)
        y_test_flat = y_test.view(y_test.shape[0], -1)

        # compute the test loss
        test_loss = loss(predictions.to(device), y_test_flat.to(device))

        # compute mean, max, and loss at quantile
        mean_loss = test_loss.mean().item()
        max_loss = test_loss.max().item()
        loss_at_quantile = torch.quantile(test_loss.cpu(), beta).item()

        satisfied_m_target = float(mean_loss <= ltt_alpha)
        satisfied_q_target = float(loss_at_quantile <= target)

        # compute mean prediction set size
        avg_set_size = predictions.sum(1).float().mean().item()
        good_ex = test_loss <= loss_at_quantile
        mean_in_quantile = test_loss[good_ex].mean().item()
        avg_quantile_set_size = predictions[good_ex].sum(1).float().mean().item()

        return TrialResult(
            mean_loss,
            loss_at_quantile,
            max_loss,
            avg_set_size,
            alpha,
            mean_in_quantile,
            avg_quantile_set_size,
            satisfied_m_target,
            satisfied_q_target,
        )

    return OrderedDict([(k, get_result(v)) for k, v in method_dict.items()])


metric_dict = {
    "balanced_acc": ("Balanced Accuracy", balanced_accuracy),
    "specificity": ("Specificity", specificity),
    "sensitivity": ("Sensitivity", sensitivity),
    "miou": ("mIoU", mIoU),
}


def main(args):
    global beta, delta
    seed_all()
    # configure experiment parameters
    num_hypotheses = args.num_hypotheses
    num_trials = args.num_trials
    save_csv = args.save_csv
    beta = args.beta

    metric_name = "miou"
    delta = 0.05

    print("Hypotheses:", num_hypotheses)
    print("Trials:", num_trials)

    def loss(y_hat: Tensor, y: Tensor) -> Tensor:
        ls = 1 - metric_dict[metric_name][1](y_hat, y)
        return ls

    # get data
    z, y = load_features_and_ground_truth("pranet", recompute=False, dev=False)
    # binarize mask
    y = y.gt(128.0).long()

    thresholds = torch.linspace(z.min().item(), z.max().item(), num_hypotheses)

    hypotheses = list(map(hypothesis_of_threshold, thresholds.tolist()))

    # configure methods
    methods = [
        VarControl,
        MinLoss,
        Ltt,
    ]
    method_dict = OrderedDict([(method.__name__, method) for method in methods])

    trial_results = [
        run_trial(
            z,
            y,
            loss,
            method_dict,
            hypotheses,
            thresholds,
            (trial_idx + 1 == num_trials),
        )
        for trial_idx in tqdm(range(num_trials))
    ]

    rows = []
    for trial_ind, trial_result in enumerate(trial_results):

        for k, v in trial_result.items():
            rows.append({"trial": trial_ind + 1, "method": k} | asdict(v))

    results_df = pd.DataFrame(rows)
    # print(results_df)
    print("\nAverage over", num_trials, "runs with", metric_name, "loss:")
    avg_results_df = (
        results_df.loc[:, results_df.columns != "trial"].groupby(["method"]).mean()
    )

    final_cols = [
        "mean_loss",
        "mean_prediction_set_size",
        "satisfied_m_target",
        "alpha",
        "quantile_loss",
        "mean_in_quantile",
        "avg_quantile_set_size",
        "satisfied_q_target",
    ]
    final_df = avg_results_df[final_cols]
    print(final_df)

    if save_csv:
        output_csv_path = (
            results_save_path
            + metric_name
            + "_beta_"
            + str(beta)
            + "_delta_"
            + str(delta)
            + ".csv"
        )
        final_df.to_csv(output_csv_path, float_format="%.4f")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("num_hypotheses", type=int, default=50)
    parser.add_argument("num_trials", type=int, default=200)
    parser.add_argument("-save_csv", type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("-beta", type=float, default=0.85)

    args = parser.parse_args()
    main(args)
