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

    # get logits and one-hot labels
    data = load_features_and_ground_truth("tresnet_m")

    print("data[0]", data[0].shape)
    print("data[1]", data[1].shape)

    # split into train and test
    (z_train, y_train), (z_test, y_test) = even_split(*data)

    # get thresholds based on min/max in logit space
    thresholds = torch.linspace(
        z_train.min().item(), z_train.max().item(), num_hypotheses
    )

    # [hyp x sample x class] get one-hot encoded vector of items present per hypothesis per sample
    hypotheses = map(hypothesis_of_threshold, thresholds.tolist())
    preds = torch.cat([h(z_train).unsqueeze(0) for h in hypotheses], 0)

    # loss per hypothesis per sample on sensitivity metric
    metric = 1 - sensitivity(preds, y_train)

    print('preds shape:', preds.shape)
    print('metric shape', metric.shape)

    delta = 0.05
    beta = 0.9
    target = 0.1
    num_train = z_train.size(0)
    inflated_beta = beta + math.sqrt(math.log(num_hypotheses / delta) / (2 * num_train))
<<<<<<< HEAD
    print("inflated beta:", inflated_beta)
=======
    print(f"inflated quantile level: {inflated_beta:0.6f}")
>>>>>>> 6856078860a38ae33c995b2209f1bfb49755410f

    emp_quantile = torch.quantile(metric, beta, dim=-1)
    inflated_quantile = torch.quantile(metric, inflated_beta, dim=-1)

    print("emp_quantile", emp_quantile.shape)

    lam = -thresholds

    plt.plot(lam, emp_quantile, "-", label=f"emp. quantile @ {beta:0.2f}")
    plt.plot(
        lam, inflated_quantile, "--", label=f"emp. quantile @ {inflated_beta:0.2f}"
    )
    plt.plot(lam, target * torch.ones_like(lam), "-.", color="black")
    plt.xlabel("$\\lambda$")
    plt.ylabel("1 - Sensitivity")
    plt.legend(loc="best")
    # plt.savefig("output/coco_var/var_plot.png", dpi=600)
