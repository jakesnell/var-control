import math
from typing import Callable
import torch
import matplotlib as mpl  # type: ignore

mpl.use("tkagg")
import matplotlib.pyplot as plt  # type: ignore

from var_control.coco_features import load_features_and_ground_truth
from var_control.metric import sensitivity
from var_control.split import even_split


def hypothesis_of_threshold(threshold: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def hypothesis(x: torch.Tensor) -> torch.Tensor:
        return x.gt(threshold).long()

    return hypothesis


if __name__ == "__main__":
    num_hypotheses = 500

    data = load_features_and_ground_truth("tresnet_m")

    (z_train, y_train), (z_test, y_test) = even_split(*data)

    thresholds = torch.linspace(
        z_train.min().item(), z_train.max().item(), num_hypotheses
    )

    hypotheses = map(hypothesis_of_threshold, thresholds.tolist())

    preds = torch.cat([h(z_train).unsqueeze(0) for h in hypotheses], 0)

    metric = 1 - sensitivity(preds, y_train).mean(-1)

    # use hoeffding's inequality to get an upper confidence bound
    delta = 0.05
    target = 0.1
    num_train = z_train.size(0)
    ucb = metric + math.sqrt(math.log(1.0 / delta) / (2 * num_train))

    lam = -thresholds

    plt.plot(lam, metric, label="sample mean")
    plt.plot(lam, ucb, "--", label=f"{int((1-delta)*100)}% UCB")
    plt.plot(lam, target * torch.ones_like(thresholds), "-.", color="black")
    plt.xlabel("$\\lambda$")
    plt.ylabel("1 - Sensitivity")
    plt.legend(loc="best")
    plt.savefig("output/coco_rcps/rcps_plot.png", dpi=600)
