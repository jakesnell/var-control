from var_control.coco_features import load_features_and_ground_truth
from var_control.metric import sensitivity


if __name__ == "__main__":
    features, y = load_features_and_ground_truth("tresnet_m")
    y_hat = features.gt(0).long()

    print(f"predictions: {y_hat}")
    print(f"ground truth: {y}")
    print(f"sensitivity: {sensitivity(y_hat, y).mean(-1).item():0.6f}")
