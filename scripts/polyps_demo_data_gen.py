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
import argparse
import pickle

from var_control.polyps_features import (
    load_features_and_ground_truth,
    load_features_demo,
)
from var_control.metric import balanced_accuracy, sensitivity, specificity
from var_control.split import fixed_split
from var_control.methods_monotonic import *

import matplotlib.pyplot as plt  # type: ignore

mpl.use("tkagg")

beta = None
delta = None
target = None
plotted = False
test_targets = torch.linspace(0.03, 1.0, 100)


task_save_path = "output/polyps/"
plot_save_path = task_save_path + "plots/sensitivity/"
results_save_path = task_save_path + "results/"


def get_predictions(hypothesis_ind, z_test, hypotheses):
    if not hypothesis_ind:
        predictions = torch.zeros_like(z_test)
    else:
        h = hypotheses[hypothesis_ind]
        predictions = h(z_test)
    return predictions


def fixed_split_with_images(
    z: torch.Tensor, y: torch.Tensor, imgs: torch.Tensor, calib_size: int
):
    num_examples = z.size(0)

    perm = torch.randperm(num_examples)
    train_perm = perm[:calib_size]
    test_perm = perm[calib_size:]

    train_set = (z[train_perm], y[train_perm], imgs[train_perm])
    test_set = (z[test_perm], y[test_perm], imgs[test_perm])

    return train_set, test_set


def run_trial(
    z: Tensor,
    y: Tensor,
    imgs: Tensor,
    loss: Callable[[Tensor, Tensor], Tensor],
    method_dict: OrderedDict,
    hypotheses: list[Hypothesis],
    thresholds: list[float],
    last_trial: bool,
):
    (z_train, y_train, img_train), (z_test, y_test, img_test) = fixed_split_with_images(
        z, y, imgs, 1000
    )
    preds = predict_batch(hypotheses, z_train)

    metric = loss(
        preds.view(preds.shape[0], preds.shape[1], -1),
        y_train.view(y_train.shape[0], -1),
    )

    results_dict = dict()
    results_dict["images"] = img_test
    results_dict["masks"] = y_test
    results_dict["predictions"] = dict()

    def get_result(method):
        global target, beta, delta, plotted
        # fit the hypothesis according to the criterion
        if method in [VarControl, Rcps]:
            hypothesis_ind = method.fit_target_var(metric, delta, beta, target)
            h = hypotheses[hypothesis_ind]
            predictions = h(z_test)
        elif method in [Ltt]:
            predictions, _ = method.fit_target_var(
                metric, z_test, target, delta, thresholds
            )

        results_dict["predictions"][method.__name__] = predictions
        predictions = predictions.view(predictions.shape[0], -1)

    for k, v in method_dict.items():
        get_result(v)

    return results_dict


metric_dict = {
    "balanced_acc": ("Balanced Accuracy", balanced_accuracy),
    "specificity": ("Specificity", specificity),
    "sensitivity": ("Sensitivity", sensitivity),
}


def main(args):
    global target, beta, delta
    # configure experiment parameters
    num_hypotheses = args.num_hypotheses
    num_trials = args.num_trials
    metric_name = "sensitivity"
    # fit the hypothesis according to the criterion
    delta = 0.05

    beta = 0.8
    target = 0.1

    print("Hypotheses:", num_hypotheses)
    print("Trials:", num_trials)

    def loss(y_hat: Tensor, y: Tensor) -> Tensor:
        ls = 1 - metric_dict[metric_name][1](y_hat, y)
        return ls

    # get data
    z, y, imgs = load_features_demo()
    # binarize mask
    y = y.gt(128.0).long()
    print("imgs", imgs.shape)

    thresholds = torch.linspace(z.min().item(), z.max().item(), num_hypotheses)

    hypotheses = list(map(hypothesis_of_threshold, thresholds.tolist()))

    # configure methods
    methods = [
        VarControl,
        Rcps,
        Ltt,
    ]
    method_dict = OrderedDict([(method.__name__, method) for method in methods])

    trial_results = run_trial(
        z, y, imgs, loss, method_dict, hypotheses, thresholds, True
    )

    print(trial_results.keys())
    print(trial_results["predictions"].keys())
    print(trial_results["images"].shape)
    print(trial_results["masks"].shape)
    print(trial_results["predictions"]["Ltt"].shape)

    with open(task_save_path + "demo_data.pkl", "wb") as handle:
        pickle.dump(trial_results, handle)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("num_hypotheses", type=int, default=50)
    parser.add_argument("num_trials", type=int, default=1)

    args = parser.parse_args()
    main(args)
