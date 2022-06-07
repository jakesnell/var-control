from collections import OrderedDict
from dataclasses import asdict, dataclass
import math
from tqdm import tqdm
from typing import Callable, TypeAlias, Tuple, Optional
import torch
from torch import Tensor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl  # type: ignore

from torch.utils.data import TensorDataset, DataLoader

from var_control.raps import *
from var_control.raps_utils import *
import pickle

from var_control.methods_monotonic import *
from var_control.nursery_features import load_features_and_ground_truth, get_label_dict
from var_control.metric import balanced_accuracy, sensitivity, specificity
from var_control.split import even_split

import matplotlib.pyplot as plt  # type: ignore

mpl.use("tkagg")


beta = None
target = None
delta = None
plotted = False
test_targets = torch.linspace(0.04, 1.0, 100)


task_save_path = "output/nursery/"
plot_save_path = task_save_path + "plots/sensitivity/"
results_save_path = task_save_path + "results/"


def get_predictions(hypothesis_ind, z_test, hypotheses):
    if not hypothesis_ind:
        predictions = torch.zeros_like(z_test)
    else:
        h = hypotheses[hypothesis_ind]
        predictions = h(z_test)
    return predictions


def plot_var_control(
    metric: Tensor,
    delta: float,
    beta: float,
    target: float,
    thresholds: list,
    group: str,
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
    plt.ylabel("1 - Sensitivity")
    plt.legend(loc="best")
    plt.savefig(
        plot_save_path + group + "_beta_" + str(beta) + "_fixed_var_plot.png", dpi=600
    )
    plt.clf()
    plt.close()


def var_plot_beta_vs_set_size(metric, delta, z_test, hypotheses, group):

    betas = torch.linspace(0.5, 0.95, 40)
    test_alphas = [0.2, 0.4, 0.6, 0.8]
    for test_alpha in test_alphas:
        set_sizes = []
        mean_losses = []
        for test_beta in betas:
            num_hypotheses = metric.size(0)
            num_train = metric.size(-1)
            inflated_beta = test_beta + math.sqrt(
                math.log(num_hypotheses / delta) / (2 * num_train)
            )
            try:
                inflated_quantile = torch.quantile(metric, inflated_beta, dim=-1)
                hypothesis_ind = int(
                    torch.nonzero(inflated_quantile <= test_alpha).view(-1).max().item()
                )
                # compute mean prediction set size
                predictions = get_predictions(hypothesis_ind, z_test, hypotheses)

                avg_set_size = predictions.sum(1).float().mean().item()
                set_sizes.append(avg_set_size)

            except Exception as e:
                continue

        plt.plot(betas[: len(set_sizes)], set_sizes, label=test_alpha)

    plt.xlabel("Beta")
    plt.ylabel("Set Size")
    # plt.legend(loc="best")
    plt.legend(title="Alphas")
    plt.title("Coverage vs. Set Size")
    plt.savefig(plot_save_path + group + "_beta_vs_set_size_sensitivity.png", dpi=600)
    plt.clf()
    plt.close()


def plot_loss_vs_coverage(test_loss, group):
    betas = torch.linspace(0.0, 1.0, 100)
    alphas = []
    for test_beta in betas:
        alpha = torch.quantile(test_loss, test_beta).item()
        alphas.append(alpha)

    plt.plot(alphas, betas)
    plt.xlabel("alpha")
    plt.ylabel("beta")
    plt.title("Emprical coverage at quantile")
    plt.savefig(plot_save_path + group + "_alpha_vs_beta_sensitivity.png", dpi=600)
    plt.clf()
    plt.close()

    plt.plot(betas, alphas)
    plt.xlabel("beta")
    plt.ylabel("alpha")
    plt.title("Emprical coverage at quantile")
    plt.savefig(plot_save_path + group + "_beta_vs_alpha_sensitivity.png", dpi=600)
    plt.clf()
    plt.close()


def run_trial(
    z: Tensor,
    y: Tensor,
    loss: Callable[[Tensor, Tensor], Tensor],
    method_dict: OrderedDict,
    hypotheses: list[Hypothesis],
    thresholds: list[float],
    group: str,
    last_trial: bool,
) -> OrderedDict[str, TrialResult]:
    global beta, delta, target
    (z_train, y_train), (z_test, y_test) = even_split(z, y)
    preds = predict_batch(hypotheses, z_train)
    metric = loss(preds, y_train)

    def get_result(method):
        global plotted
        # fit the hypothesis according to the criterion
        if method in [VarControl, Rcps]:
            hypothesis_ind = method.fit_target_var(metric, delta, beta, target)
            h = hypotheses[hypothesis_ind]
            predictions = h(z_test)

        elif method in [ConformalPred, Aps, Raps]:
            predictions = method.fit_target_var(z_train, y_train, z_test, target)
        elif method in [Ltt]:
            predictions, _ = method.fit_target_var(
                metric, z_test, target, delta, thresholds
            )

        if last_trial and not plotted:

            plot_var_control(metric, delta, beta, target, thresholds, group)
            var_plot_beta_vs_set_size(metric, delta, z_train, hypotheses, group)
            plotted = True

        # compute the test loss
        test_loss = loss(predictions, y_test)

        # if last_trial and method == VarControl:
        #     plot_loss_vs_coverage(test_loss, group)

        # compute mean, max, and loss at quantile
        mean_loss = test_loss.mean().item()
        max_loss = test_loss.max().item()
        loss_at_quantile = torch.quantile(test_loss, beta).item()

        satisfied_m_target = float(mean_loss <= target)
        satisfied_q_target = float(loss_at_quantile <= target)

        # compute mean prediction set size
        avg_set_size = predictions.sum(1).float().mean().item()

        good_ex = test_loss <= loss_at_quantile
        mean_in_quantile = test_loss[good_ex].mean().item()
        avg_quantile_set_size = (
            predictions[good_ex.squeeze(0)].sum(1).float().mean().item()
        )

        return TrialResult(
            mean_loss,
            loss_at_quantile,
            max_loss,
            avg_set_size,
            mean_in_quantile,
            avg_quantile_set_size,
            satisfied_m_target,
            satisfied_q_target,
        )

    return OrderedDict([(k, get_result(v)) for k, v in method_dict.items()])


metric_dict = {
    "balanced_acc": ("Balanced Accuracy", balanced_accuracy),  # 0.5 * (sens+spec)
    "specificity": ("Specificity", specificity),  # are we making too many predictions?
    "sensitivity": (
        "Sensitivity",
        sensitivity,
    ),  # is ground truth included in prediction sets?
}


def main():
    global beta, delta, target, plotted
    # configure experiment parameters
    num_hypotheses = 500
    num_trials = 2
    metric_name = "sensitivity"
    loss_type = "fixed"
    beta = 0.9
    target = 0.1
    delta = 0.05
    feature_key = "logreg"

    def weighted_loss(y_hat: Tensor, y: Tensor) -> Tensor:

        n = y.shape[0]
        # compute unweighted loss
        fixed_ls = 1 - metric_dict[metric_name][1](y_hat, y)
        # convert one-hot matrix to label vector
        y_labels = torch.argmax(y, dim=1).tolist()
        # get vector of per sample loss weights
        weight_per_sample = label_weights[y_labels].unsqueeze(0)
        # apply class specific loss to each sample
        ls = (fixed_ls > 0.0).float().view(-1, n) * weight_per_sample

        return ls

    def fixed_loss(y_hat: Tensor, y: Tensor) -> Tensor:
        ls = 1 - metric_dict[metric_name][1](y_hat, y)
        return ls

    if loss_type == "fixed":
        loss = fixed_loss
    elif loss_type == "weighted":
        label_weights = torch.tensor([1.0, 0.8, 0.6, 0.6, 1.0])
        # label_weights = torch.tensor([1.25, 1.0, 1.0, 1.0, 1.25])
        loss = weighted_loss
    # get data
    z_p, y_p, z_n, y_n = load_features_and_ground_truth(feature_key)
    label_dict = get_label_dict()

    split_sets = [
        (z_n, y_n, label_dict["finance"][0]),
        (z_p, y_p, label_dict["finance"][1]),
    ]
    rows = []
    for z, y, group in split_sets:
        plotted = False

        thresholds = torch.linspace(z.min().item(), z.max().item(), num_hypotheses)

        hypotheses = list(map(hypothesis_of_threshold, thresholds.tolist()))

        # configure methods
        methods = [
            VarControl,
            Rcps,
            Ltt,
            Raps,
            Aps,
            ConformalPred,
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
                group,
                (trial_idx + 1 == num_trials),
            )
            for trial_idx in tqdm(range(num_trials))
        ]

        for trial_ind, trial_result in enumerate(trial_results):
            for k, v in trial_result.items():
                rows.append(
                    {"trial": trial_ind + 1, "method": k, "group": group} | asdict(v)
                )

    results_df = pd.DataFrame(rows)
    print("Average over", num_trials, "runs with", loss_type, "loss:")
    avg_results_df = (
        results_df.loc[:, results_df.columns != "trial"]
        .groupby(["method", "group"])
        .mean()
    )

    final_cols = [
        "mean_loss",
        "mean_prediction_set_size",
        "satisfied_m_target",
        "quantile_loss",
        "mean_in_quantile",
        "avg_quantile_set_size",
        "satisfied_q_target",
    ]
    final_df = avg_results_df[final_cols]

    print("  alpha:", target, "| beta:", beta, "| delta:", delta)
    print(final_df)

    output_csv_path = (
        results_save_path
        + loss_type
        + "_beta_"
        + str(beta)
        + "_target_"
        + str(target)
        + "_delta_"
        + str(delta)
        + ".csv"
    )
    print(output_csv_path)
    final_df.to_csv(output_csv_path, float_format="%.4f")


if __name__ == "__main__":
    main()
