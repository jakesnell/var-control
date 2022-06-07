import torch
from torchtyping import TensorType  # type: ignore
from torchtyping import patch_typeguard
from typeguard import typechecked
import pickle
from os import listdir
from os.path import isfile, join
from os.path import exists
import numpy as np
from skimage.transform import resize

batch = None
classes = None
size = None

patch_typeguard()

polyp_features_save_name = "data/polyps/pranet_features.pkl"


def load_features_demo(
        data_path: str = "data/polyps/",
        dev: bool = True
):
    devsize = 64
    testsize = 352

    z = []
    y = []
    imgs = []
    for _data_name in ['HyperKvasir', 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

        save_path = data_path + '{}/'.format(_data_name)
        onlyfiles = [f for f in listdir(save_path) if isfile(join(save_path, f))]

        for file in onlyfiles:
            data = np.load(save_path + file)
            scores = data['arr_0']
            mask = data['arr_1']
            img = data['arr_2'][0]

            if dev:
                resized_scores = resize(scores, (devsize, devsize), anti_aliasing=False)
                resized_mask = resize(mask, (devsize, devsize), anti_aliasing=False)
                resized_img = resize(img, (3, devsize, devsize), anti_aliasing=False)
                z.append(resized_scores)
                y.append(resized_mask)
                imgs.append(resized_img)
            else:
                resized_mask = resize(mask, (testsize, testsize), anti_aliasing=False)
                z.append(scores)
                y.append(resized_mask)
                imgs.append(img)

    z = torch.tensor(np.array(z))
    y = torch.tensor(np.array(y))
    imgs = torch.tensor(np.array(imgs))

    return z, y, imgs


@typechecked
def load_features(
        data_path: str,
        recompute: bool = False,
        dev: bool = False
) -> tuple[TensorType["batch", "size", "size", float], TensorType["batch", "size", "size", float]]:
    if exists(polyp_features_save_name) and not recompute:
        print("Loading polyps features")
        with open(polyp_features_save_name, 'rb') as handle:
            z, y = pickle.load(handle)
    else:
        print("Computing polyps features")
        testsize = 352
        devsize = 28
        z = []
        y = []
        for _data_name in ['HyperKvasir', 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

            save_path = data_path + '{}/'.format(_data_name)
            onlyfiles = [f for f in listdir(save_path) if isfile(join(save_path, f))]

            for file in onlyfiles:
                data = np.load(save_path + file)
                scores = data['arr_0']
                mask = data['arr_1']

                if dev:
                    resized_scores = resize(scores, (devsize, devsize), anti_aliasing=False)
                    resized_mask = resize(mask, (devsize, devsize), anti_aliasing=False)
                    z.append(resized_scores)
                    y.append(resized_mask)
                else:
                    resized_mask = resize(mask, (testsize, testsize), anti_aliasing=False)
                    z.append(scores)
                    y.append(resized_mask)

        z = torch.tensor(np.array(z))
        y = torch.tensor(np.array(y))

        with open(polyp_features_save_name, 'wb') as handle:
            pickle.dump((z, y), handle)

    return z, y


@typechecked
def load_features_and_ground_truth(
        feature_key: str,
        recompute: bool = False,
        dev: bool = False,
) -> tuple[TensorType[float], TensorType[float]]:
    match feature_key:
        case "pranet":
            data_path = "data/polyps/"
        case _:
            raise ValueError(f"unknown feature_key: {feature_key}")
    predictions, masks = load_features(data_path, recompute=recompute, dev=dev)
    return predictions, masks

