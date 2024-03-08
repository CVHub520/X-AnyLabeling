import logging

import cv2
import numpy as np
from .general import refine_contours


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

    Args:
        x (np.ndarray): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
       y (np.ndarray): The bounding box coordinates in (x, y, width, height) format.
    """
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        x (np.ndarray): The bounding box coordinates.
        w (int): Width of the image. Defaults to 640
        h (int): Height of the image. Defaults to 640
        padw (int): Padding width. Defaults to 0
        padh (int): Padding height. Defaults to 0
    Returns:
        y (np.ndarray): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
            x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
    """
    y = np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format.
    x, y, width and height are normalized to image dimensions

    Args:
        x (np.ndarray): The input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): The width of the image. Defaults to 640
        h (int): The height of the image. Defaults to 640
        clip (bool): If True, the boxes will be clipped to the image boundaries. Defaults to False
        eps (float): The minimum value of the box's width and height. Defaults to 0.0
    Returns:
        y (np.ndarray): The bounding box coordinates in (x, y, width, height, normalized) format
    """
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    """
    Convert normalized coordinates to pixel coordinates of shape (n,2)

    Args:
        x (np.ndarray): The input tensor of normalized bounding box coordinates
        w (int): The width of the image. Defaults to 640
        h (int): The height of the image. Defaults to 640
        padw (int): The width of the padding. Defaults to 0
        padh (int): The height of the padding. Defaults to 0
    Returns:
        y (np.ndarray): The x and y coordinates of the top left corner of the bounding box
    """
    y = np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y


def xywh2ltwh(x):
    """
    Convert the bounding box format from [x, y, w, h] to [x1, y1, w, h], where x1, y1 are the top-left coordinates.

    Args:
        x (np.ndarray): The input tensor with the bounding box coordinates in the xywh format
    Returns:
        y (np.ndarray): The bounding box coordinates in the xyltwh format
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    return y


def xyxy2ltwh(x):
    """
    Convert nx4 bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h], where xy1=top-left, xy2=bottom-right

    Args:
      x (np.ndarray): The input tensor with the bounding boxes coordinates in the xyxy format
    Returns:
      y (np.ndarray): The bounding box coordinates in the xyltwh format.
    """
    y = np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def ltwh2xywh(x):
    """
    Convert nx4 boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center

    Args:
      x (np.ndarray): the input tensor
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] + x[:, 2] / 2  # center x
    y[:, 1] = x[:, 1] + x[:, 3] / 2  # center y
    return y


def ltwh2xyxy(x):
    """
    It converts the bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right

    Args:
      x (np.ndarray): the input image

    Returns:
      y (np.ndarray): the xyxy coordinates of the bounding boxes.
    """
    y = np.copy(x)
    y[:, 2] = x[:, 2] + x[:, 0]  # width
    y[:, 3] = x[:, 3] + x[:, 1]  # height
    return y


def cxywh2xyxy(x):
    """
    Converts bounding box coordinates from [cx, cy, w, h] to [x1, y1, x2, y2] format.

    Args:
        x (np.ndarray): Input bounding box coordinates in the format [cx, cy, w, h].

    Returns:
        y (np.ndarray): Converted bounding box coordinates in the format [x1, y1, x2, y2].
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - 0.5 * x[:, 2]  # x1
    y[:, 1] = x[:, 1] - 0.5 * x[:, 3]  # y1
    y[:, 2] = x[:, 0] + 0.5 * x[:, 2]  # x2
    y[:, 3] = x[:, 1] + 0.5 * x[:, 3]  # y2
    return y


def xywhr2xyxyxyxy(center):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in degrees from 0 to 90.

    Args:
        center (numpy.ndarray): Input data in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin = (np.cos, np.sin)

    ctr = center[..., :2]
    w, h, angle = (center[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1)
    vec2 = np.concatenate(vec2, axis=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2)


def rbox2poly(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
    """
    center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, -w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, -h / 2 * Cos], axis=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    order = obboxes.shape[:-1]
    return np.concatenate([point1, point2, point3, point4], axis=-1).reshape(
        *order, 8
    )


def denormalize_bbox(bbox, input_shape, image_shape):
    """
    Denormalizes bounding box coordinates from input_shape to image_shape.

    Parameters:
    - bbox: Normalized bounding box coordinates [xmin, ymin, xmax, ymax]
    - input_shape: The shape of the input image used during normalization (e.g., [640, 640])
    - image_shape: The shape of the original image (e.g., [height, width])

    Returns:
    - Denormalized bounding box coordinates [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox

    # Denormalize x-coordinates
    denorm_xmin = int(xmin * image_shape[1] / input_shape[1])
    denorm_xmax = int(xmax * image_shape[1] / input_shape[1])

    # Denormalize y-coordinates
    denorm_ymin = int(ymin * image_shape[0] / input_shape[0])
    denorm_ymax = int(ymax * image_shape[0] / input_shape[0])

    denormalized_bbox = [denorm_xmin, denorm_ymin, denorm_xmax, denorm_ymax]

    return denormalized_bbox


def rescale_box(input_shape, boxes, image_shape, kpts=False):
    """Rescale the output to the original image shape"""
    ratio = min(
        input_shape[0] / image_shape[0],
        input_shape[1] / image_shape[1],
    )
    padding = (
        (input_shape[1] - image_shape[1] * ratio) / 2,
        (input_shape[0] - image_shape[0] * ratio) / 2,
    )
    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio
    boxes[:, 0] = np.clip(boxes[:, 0], 0, image_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, image_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, image_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, image_shape[0])  # y2
    if kpts:
        num_kpts = boxes.shape[1] // 3
        for i in range(2, num_kpts + 1):
            boxes[:, i * 3 - 1] = (boxes[:, i * 3 - 1] - padding[0]) / ratio
            boxes[:, i * 3] = (boxes[:, i * 3] - padding[1]) / ratio
    return boxes


def rescale_box_and_landmark(input_shape, boxes, lmdks, image_shape):
    ratio = min(
        input_shape[0] / image_shape[0],
        input_shape[1] / image_shape[1],
    )
    padding = (
        (input_shape[1] - image_shape[1] * ratio) / 2,
        (input_shape[0] - image_shape[0] * ratio) / 2,
    )
    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio
    boxes[:, 0] = np.clip(boxes[:, 0], 0, image_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, image_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, image_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, image_shape[0])  # y2
    # lmdks
    lmdks[:, [0, 2, 4, 6, 8]] -= padding[0]
    lmdks[:, [1, 3, 5, 7, 9]] -= padding[1]
    lmdks[:, :10] /= ratio
    lmdks[:, 0] = np.clip(lmdks[:, 0], 0, image_shape[1])
    lmdks[:, 1] = np.clip(lmdks[:, 1], 0, image_shape[0])
    lmdks[:, 2] = np.clip(lmdks[:, 2], 0, image_shape[1])
    lmdks[:, 3] = np.clip(lmdks[:, 3], 0, image_shape[0])
    lmdks[:, 4] = np.clip(lmdks[:, 4], 0, image_shape[1])
    lmdks[:, 5] = np.clip(lmdks[:, 5], 0, image_shape[0])
    lmdks[:, 6] = np.clip(lmdks[:, 6], 0, image_shape[1])
    lmdks[:, 7] = np.clip(lmdks[:, 7], 0, image_shape[0])
    lmdks[:, 8] = np.clip(lmdks[:, 8], 0, image_shape[1])
    lmdks[:, 9] = np.clip(lmdks[:, 9], 0, image_shape[0])

    return np.round(boxes), np.round(lmdks)


def rescale_tlwh(input_shape, boxes, image_shape, kpts=False):
    """Rescale the output to the original image shape"""
    ratio = min(
        input_shape[0] / image_shape[0],
        input_shape[1] / image_shape[1],
    )
    padding = (
        (input_shape[1] - image_shape[1] * ratio) / 2,
        (input_shape[0] - image_shape[0] * ratio) / 2,
    )
    boxes[:, 0] -= padding[0]
    boxes[:, 1] -= padding[1]
    boxes[:, :4] /= ratio
    boxes[:, 0] = np.clip(boxes[:, 0], 0, image_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, image_shape[0])  # y1
    boxes[:, 2] = np.clip((boxes[:, 0] + boxes[:, 2]), 0, image_shape[1])  # x2
    boxes[:, 3] = np.clip((boxes[:, 1] + boxes[:, 3]), 0, image_shape[0])  # y2
    if kpts:
        num_kpts = boxes.shape[1] // 3
        for i in range(2, num_kpts + 1):
            boxes[:, i * 3 - 1] = (boxes[:, i * 3 - 1] - padding[0]) / ratio
            boxes[:, i * 3] = (boxes[:, i * 3] - padding[1]) / ratio
    return boxes


def scale_boxes(
    img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False
):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
      img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
      boxes (np.ndarray): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
      img0_shape (tuple): the shape of the target image, in the format of (height, width).
      ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                         calculated based on the size difference between the two images.
      padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
        rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
      boxes (np.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0]] -= pad[0]  # x padding
        boxes[..., [1]] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_masks(masks, shape, padding=True):
    """
    Rescale segment masks to shape.

    Args:
        masks (np.ndarray): (C, H, W).
        shape (tuple): Height and width with input shape.
        padding (bool): If True, assuming the boxes are based on an image augmented by YOLO style.
                        If False, then do regular rescaling.
    """
    _, mh, mw = masks.shape
    gain = min(mh / shape[0], mw / shape[1])  # gain = old / new
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # y, x
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[:, top:bottom, left:right]
    # Resizing without loop
    masks = cv2.resize(
        masks.transpose((1, 2, 0)),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    masks = masks.transpose((2, 0, 1))
    return masks


def scale_coords(
    img1_shape,
    coords,
    img0_shape,
    ratio_pad=None,
    normalize=False,
    padding=True,
):
    """
    Rescale segment coordinates (xyxy) from img1_shape to img0_shape

    Args:
      img1_shape (tuple): The shape of the image that the coords are from.
      coords (np.ndarray): the coords to be scaled
      img0_shape (tuple): the shape of the image that the segmentation is being applied to
      ratio_pad (tuple): the ratio of the image size to the padded image size.
      normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False
      padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
        rescaling.

    Returns:
      coords (np.ndarray): the segmented image.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (None): The function modifies the input `coordinates` in place, by clipping each coordinate to the image boundaries.
    """
    coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
    coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y


def clip_boxes(boxes, shape):
    """
    It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape

    Args:
      boxes (np.ndarray): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def masks2segments(masks, epsilon_factor=0):
    """
    It takes a list of masks(n,h,w) and returns a list of segments(n,xy)

    Args:
      masks (np.ndarray):
        the output of the model, which is a tensor of shape (batch_size, 160, 160)
      epsilon_factor (float, optional):
        Factor used for epsilon calculation in contour approximation.
        A smaller value results in smoother contours but with more points.
        If the value is set to 0, the default result will be used.

    Returns:
      segments (List): list of segment masks
    """
    segments = []
    for x in masks.astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        img_area = masks.shape[1] * masks.shape[2]
        c = refine_contours(c, img_area, epsilon_factor)
        if c:
            c = np.array([c[0] for c in c[0]])
            c = np.concatenate([c, [c[0]]])  # Close the contour
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments


def tlwh_to_xyxy(x):
    """ " Convert tlwh to xyxy"""
    x1 = x[0]
    y1 = x[1]
    x2 = x[2] + x1
    y2 = x[3] + y1
    return [x1, y1, x2, y2]


def xyxy_to_tlwh(x):
    x1, y1, x2, y2 = x
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]
