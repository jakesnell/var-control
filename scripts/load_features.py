from var_control.coco_features import load_features_and_ground_truth

if __name__ == "__main__":
    features, ground_truth = load_features_and_ground_truth("tresnet_m")
    print(f"features: {features}\nground truth: {ground_truth}")
