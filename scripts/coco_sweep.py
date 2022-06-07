import argparse
from argparse import Namespace
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

from var_control.coco_features import load_features_and_ground_truth
from var_control.methods import VarControl
from var_control.metric import sensitivity, balanced_accuracy
from var_control.split import even_split_iterator

from coco_experiments import Batch, predict
from scripts.coco_experiments import run_monotonic_trial


def run_monotonic_sweep(train_batch: Batch, test_batch: Batch, delta: float):
    for alpha in [0.1, 0.2, 0.3, 0.4]:
        beta = torch.linspace(0, 0.9, 51)
        var_results = [
            VarControl.fit_target_var(train_batch.loss, delta, b.item(), alpha)
            for b in beta
        ]
        pred_size = torch.Tensor(
            [test_batch.pred_size[ind].float().mean(-1).item() for ind in var_results]
        )
        plt.plot(beta, pred_size, "-o", label=f"alpha = {alpha}")
    plt.legend(loc="best")
    plt.xlabel("beta")
    plt.ylabel("mean pred. set size")
    plt.title("MS-COCO VarControl on Sensitivity")
    plt.savefig("output/coco_sweep/monotonic.png", bbox_inches="tight")


def run_general_sweep(train_batch: Batch, test_batch: Batch, delta: float):
    beta = torch.linspace(0, 0.9, 51)
    var_results = [VarControl.fit_var(train_batch.loss, delta, b.item()) for b in beta]
    alpha = torch.Tensor([a for (_, a) in var_results])
    plt.plot(beta, alpha, "-o")
    plt.xlabel("beta")
    plt.ylabel("alpha")
    plt.title("MS-COCO VarControl on Balanced Acc.")
    plt.savefig("output/coco_sweep/general.png", bbox_inches="tight")


def main(args: Namespace):
    torch.manual_seed(args.seed)

    if args.loss == "sensitivity":
        loss_fn = lambda pred, y: 1 - sensitivity(pred, y)
        loss_type = "monotonic"
    elif args.loss == "balanced_accuracy":
        loss_fn = lambda pred, y: 1 - balanced_accuracy(pred, y)
        loss_type = "general"
    else:
        raise ValueError(f"Unexpected loss {args.loss}")

    # y: [N, C]
    z, y = load_features_and_ground_truth("tresnet_m")

    thresholds = torch.linspace(z.min().item(), z.max().item(), args.num_hypotheses)

    # make predictions on the entire set
    preds = predict(thresholds, z).permute(1, 0, 2)  # [N, H, C] (long)

    loss = loss_fn(preds, y.unsqueeze(1))

    for train_sample, test_sample in even_split_iterator(
        TensorDataset(preds.sum(-1), loss), 1
    ):
        train_batch = Batch([x.t() for x in train_sample], args.num_hypotheses)
        test_batch = Batch([x.t() for x in test_sample], args.num_hypotheses)

    if loss_type == "monotonic":
        run_monotonic_sweep(train_batch, test_batch, args.delta)
    else:
        run_general_sweep(train_batch, test_batch, args.delta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run sweep over alpha for COCO experiments"
    )
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
        "--delta",
        type=float,
        default=0.05,
        help="acceptable probability of error (default: 0.05)",
    )
    args = parser.parse_args()
    main(args)
