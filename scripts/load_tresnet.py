from __future__ import annotations
import argparse
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.jit._script import ScriptModule


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a pretrained tresnet model and display its output on a sample image.",
    )
    parser.add_argument("size", help="tresnet size (options are 'm', 'l', or 'xl')")
    parser.add_argument(
        "-i",
        "--image_path",
        help="path to sample image",
        default="data/sample_images/tennis.jpg",
    )
    return parser


@dataclass(frozen=True)
class TresnetArgs:
    path: str
    image_size: int


def load_tresnet_args(size: str) -> TresnetArgs:
    match size:
        case "m":
            image_size = 224
        case "l":
            image_size = 448
        case "xl":
            image_size = 640
        case _:
            raise ValueError(f"could not find tresnet args for {size}")

    return TresnetArgs(f"models/tresnet/tresnet_{size}.pt", image_size)


def load_tresnet(size: str) -> tuple[ScriptModule, int]:
    tresnet_args = load_tresnet_args(size)
    return (
        torch.load(tresnet_args.path, map_location=torch.device("cpu")),
        tresnet_args.image_size,
    )


def load_image(pic_path: str, image_size: int) -> torch.Tensor:
    im = Image.open(pic_path)
    im_resize = im.resize((image_size, image_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    return torch.unsqueeze(tensor_img, 0)


def main() -> None:
    args = init_argparse().parse_args()
    model, image_size = load_tresnet(args.size)
    tensor_batch = load_image(args.image_path, image_size)
    output = model(tensor_batch)
    print(output)


if __name__ == "__main__":
    main()
