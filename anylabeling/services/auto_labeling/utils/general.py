import cv2
import numpy as np


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def letterbox(
    im,
    new_shape,
    color=(114, 114, 114),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """
    Resize and pad image while meeting stride-multiple constraints
    Returns:
        im (array): (height, width, 3)
        ratio (array): [w_ratio, h_ratio]
        (dw, dh) (array): [w_padding h_padding]
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):  # [h_rect, w_rect]
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # wh ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w h
    dw, dh = (
        new_shape[1] - new_unpad[0],
        new_shape[0] - new_unpad[1],
    )  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])  # [w h]
        ratio = (
            new_shape[1] / shape[1],
            new_shape[0] / shape[0],
        )  # [w_ratio, h_ratio]

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)


def softmax(x):
    """
    Applies the softmax function to the input array.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output array after applying softmax.
    """
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def refine_contours(contours, img_area, epsilon_factor=0.001):
    """
    Refine contours by approximating and filtering.

    Parameters:
    - contours (list): List of input contours.
    - img_area (int): Maximum factor for contour area.
    - epsilon_factor (float, optional): Factor used for epsilon calculation in contour approximation. Default is 0.001.

    Returns:
    - list: List of refined contours.
    """
    # Refine contours
    approx_contours = []
    for contour in contours:
        # Approximate contour
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx_contours.append(approx)

    # Remove too big contours ( >90% of image size)
    if len(approx_contours) > 1:
        areas = [cv2.contourArea(contour) for contour in approx_contours]
        filtered_approx_contours = [
            contour
            for contour, area in zip(approx_contours, areas)
            if area < img_area * 0.9
        ]

    # Remove small contours (area < 20% of average area)
    if len(approx_contours) > 1:
        areas = [cv2.contourArea(contour) for contour in approx_contours]
        avg_area = np.mean(areas)

        filtered_approx_contours = [
            contour
            for contour, area in zip(approx_contours, areas)
            if area > avg_area * 0.2
        ]
        approx_contours = filtered_approx_contours

    return approx_contours
