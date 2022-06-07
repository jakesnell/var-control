import math
from typing import Callable
import torch
import matplotlib as mpl  # type: ignore

mpl.use("tkagg")
import matplotlib.pyplot as plt  # type: ignore

from var_control.coco_features import load_features_and_ground_truth
from var_control.metric import balanced_accuracy
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

    hypotheses = list(map(hypothesis_of_threshold, thresholds.tolist()))

    preds = torch.cat([h(z_train).unsqueeze(0) for h in hypotheses], 0)

    metric = 1 - balanced_accuracy(preds, y_train)

    delta = 0.05
    beta = 0.9
    num_train = z_train.size(0)
    inflated_beta = beta + math.sqrt(math.log(num_hypotheses / delta) / (2 * num_train))
    print(f"inflated quantile level: {inflated_beta:0.6f}")

    emp_quantile = torch.quantile(metric, beta, dim=-1)
    inflated_quantile = torch.quantile(metric, inflated_beta, dim=-1)

    lam = -thresholds

    plt.plot(lam, emp_quantile, "-", label=f"emp. quantile @ {beta:0.2f}")
    plt.plot(
        lam, inflated_quantile, "--", label=f"emp. quantile @ {inflated_beta:0.2f}"
    )
    plt.xlabel("$\\lambda$")
    plt.ylabel("1 - Balanced Accuracy")
    plt.legend(loc="best")
    plt.savefig("output/coco_var_balanced_accuracy/balanced_accuracy_plot.png", dpi=600)

    best_ind = torch.argmin(inflated_quantile)
    best_loss = inflated_quantile[best_ind]
    best_hypothesis = hypotheses[int(best_ind.item())]
    print(f"loss threshold: {best_loss.item():0.6f}")

    test_loss_at_beta = torch.quantile(
        1 - balanced_accuracy(best_hypothesis(z_test), y_test), beta, dim=-1
    )

    print(f"test loss at beta: {test_loss_at_beta:0.6f}")
