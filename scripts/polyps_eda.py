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

from var_control.polyps_features import load_features_and_ground_truth
from var_control.metric import balanced_accuracy, sensitivity, specificity
from var_control.split import fixed_split
from var_control.methods_monotonic import *

import matplotlib.pyplot as plt  # type: ignore

z, y = load_features_and_ground_truth("pranet", recompute=False, dev=False)
print(y.shape)
y = y.gt(128.0).long()
avg_size = y.sum(-1).sum(-1).float().mean()
print(avg_size)
