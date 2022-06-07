#Var-Control

Code for paper "VaR-Control: Bounding the Probability of High-Loss Predictions"

Our method provides distribution-free guarantees for loss @ quantiles.

To initialize the conda environment, run the following commands:

    make env-init
    conda activate var_control

Requires python >= 3.10

### Source Data and Models

#### Tiny-ImageNet

- Tiny-ImageNet is available for download at http://cs231n.stanford.edu/tiny-imagenet-200.zip
- Tiny-ImageNet-c is available for download at https://zenodo.org/record/2469796#.YpjtGWDMJPs
- Our model is a ResNet50 fine-tuned on the training set of Tiny-ImageNet

#### MS-COCO

- MS-COCO is available for download at https://cocodataset.org/
- The TResNet model can be obtained at https://github.com/Alibaba-MIIL/TResNet

#### Polyps

- Test images and ground truth for Kvasir, ETIS-LaribPolypDB, and the CVC 
datasets can be downloaded via the PraNet github https://github.com/DengPingFan/PraNet
- Test images and ground truth for HyperKvasir can be downloaded at https://datasets.simula.no/hyper-kvasir/
- PraNet model is available via github https://github.com/DengPingFan/PraNet

#### Nursery

- Dataset can be downloaded from https://archive.ics.uci.edu/ml/datasets/nursery

### Running Experiments

Run the below commands with appropriate parameters to reproduce our experiments

#### Tiny-ImageNet class-varying recall

    python scripts/tiny_imagenet_monotonic.py -mixed=MIXED -beta=BETA -target=TARGET

Mixed is a boolean value representing whether or not to mix in images from Tiny-ImageNet-c
#### Tiny-ImageNet balanced accuracy

    python scripts/tiny_imagenet_non_monotonic.py -mixed=MIXED -beta=BETA

Mixed is a boolean value representing whether or not to mix in images from Tiny-ImageNet-c

#### MS-COCO sensitivity

    python scripts/coco_experiments.py sensitivity

#### MS-COCO balanced accuracy

    python scripts/coco_experiments.py balanced_accuracy

#### Polyps sensitivity

    python scripts/polyps_monotonic.py -beta BETA -target=TARGET

#### Polyps IoU

    python scripts/polyps_non_monotonic.py -beta=BETA

#### Nursery (Balanced and Weighted Accuracy)

    python scripts/nursery_non_monotonic.py -beta=BETA -loss=LOSS

Loss can be either 'weighted' or 'fixed'
