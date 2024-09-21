# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.


import os

import numpy as np

from anylabeling.services.auto_labeling.utils.sahi.utils.import_utils import (
    is_available,
)

if is_available("torch"):
    import torch
else:
    torch = None


def empty_cuda_cache():
    if is_torch_cuda_available():
        return torch.cuda.empty_cache()


def to_float_tensor(img):
    """
    Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W).
    Args:
        img: np.ndarray
    Returns:
        torch.tensor
    """

    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(np.array(img)).float()
    if img.max() > 1:
        img /= 255

    return img


def torch_to_numpy(img):
    img = img.numpy()
    if img.max() > 1:
        img /= 255
    return img.transpose((1, 2, 0))


def is_torch_cuda_available():
    if is_available("torch"):
        return torch.cuda.is_available()
    else:
        return False


def select_device(device: str):
    """
    Selects torch device

    Args:
        device: str
            "cpu", "mps", "cuda", "cuda:0", "cuda:1", etc.

    Returns:
        torch.device

    Inspired by https://github.com/ultralytics/yolov5/blob/6371de8879e7ad7ec5283e8b95cc6dd85d6a5e72/utils/torch_utils.py#L107
    """
    if device == "cuda":
        device = "cuda:0"
    device = (
        str(device).strip().lower().replace("cuda:", "").replace("none", "")
    )  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            "-1"  # force torch.cuda.is_available() = False
        )
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            device  # set environment variable - must be before assert is_available()
        )

    if (
        not cpu and not mps and is_torch_cuda_available()
    ):  # prefer GPU if available
        arg = "cuda:0"
    elif (
        mps
        and getattr(torch, "has_mps", False)
        and torch.backends.mps.is_available()
    ):  # prefer MPS if available
        arg = "mps"
    else:  # revert to CPU
        arg = "cpu"

    return torch.device(arg)
