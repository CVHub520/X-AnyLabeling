import os
import cv2
import time as time
import numpy as np
import onnxruntime as ort

"""
The onnxruntime demo of the YOLOv8-OBB
Written by Wei Wang (CVHub)
    Usage:
        1. git clone https://github.com/ultralytics/ultralytics
        2. cd ultralytics and pip install -r requirements.txt
        3. export PYTHONPATH=/path/to/your/ultralytics
        4. Place the current script in this directory
        5. Download the corresponding weights
        6. Export the model to onnx format:
        ```python
        from ultralytics import YOLO
        model = YOLO("path/to/yolov8/obb/yolov8s-obb.pt")
        success = model.export(format="onnx", simplify=True)
        ```
        7. Modified the values and run this script
        ```bash
        python ${export_yolov8_obb_onnx.py}
        ```
"""


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


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1, box2):
    area1 = box_area(box1)  # N
    area2 = box_area(box2)  # M
    # broadcasting
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou  # NxM


def numpy_nms(boxes, scores, iou_threshold):
    idxs = scores.argsort()
    keep = []
    while idxs.size > 0:
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]
        other_boxes = boxes[idxs]
        ious = box_iou(max_score_box, other_boxes)
        idxs = idxs[ious[0] <= iou_threshold]
    keep = np.array(keep)
    return keep


def numpy_nms_rotated(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)

    sorted_idx = np.argsort(scores)[::-1]
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes)
    ious = np.triu(ious, k=1)
    pick = np.nonzero(np.max(ious, axis=0) < iou_threshold)[0]
    return sorted_idx[pick]


def batch_probiou(obb1, obb2, eps=1e-7):
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in np.split(obb2[..., :2], 2, axis=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))
    t1 = (
        (
            (a1 + a2) * (np.power(y1 - y2, 2))
            + (b1 + b2) * (np.power(x1 - x2, 2))
        )
        / ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2))
        / ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)) + eps)
    ) * 0.5

    t3 = (
        np.log(
            ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)))
            / (
                4
                * np.sqrt(
                    (a1 * b1 - np.power(c1, 2)).clip(0)
                    * (a2 * b2 - np.power(c2, 2)).clip(0)
                )
                + eps
            )
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = np.clip(bd, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def _get_covariance_matrix(boxes):
    gbbs = np.concatenate(
        (np.power(boxes[:, 2:4], 2) / 12, boxes[:, 4:]), axis=-1
    )
    a, b, c = np.split(gbbs, [1, 2], axis=-1)
    return (
        a * np.cos(c) ** 2 + b * np.sin(c) ** 2,
        a * np.sin(c) ** 2 + b * np.cos(c) ** 2,
        a * np.cos(c) * np.sin(c) - b * np.sin(c) * np.cos(c),
    )


def non_max_suppression_v8(
    prediction,
    task="det",
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_nms=30000,
    max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, \
        with support for masks and multiple labels per box.

    Arguments:
        prediction (np.array):
            A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks.
            The tensor should be in the format output by a model, such as YOLO.
        task: `det` | `seg` | `track` | `obb`
        conf_thres (float):
            The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float):
            The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider.
            If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes,
            and all classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, np.array]]]):
            A list of lists, where each inner list contains the apriori labels \
            for a given image. The list should be in the format output by a dataloader, \
            with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. \
            Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into numpy_nms.
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[np.array]):
            A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes,
            with columns (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, \
        valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, \
        valid values are between 0.0 and 1.0"
    if task == "seg" and nc == 0:
        raise ValueError("The value of nc must be set when the mode is 'seg'.")
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output
    bs = prediction.shape[0]  # batch size
    if task in ["det", "track"]:
        nc = prediction.shape[1] - 4  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres  # candidates

    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    # shape(1,84,6300) to shape(1,6300,84)
    prediction = np.transpose(prediction, (0, 2, 1))
    # if task != "obb":
    #     prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
    output = [np.zeros((0, 6 + nm))] * bs

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) |
        # (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        if labels and len(labels[xi]) and task != "obb":
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[np.arange(len(lb)), lb[:, 0].astype(int) + 4] = 1.0  # cls
            x = np.concatenate((x[xc], v), axis=0)

        if not x.shape[0]:
            continue

        box = x[:, :4]
        cls = x[:, 4 : 4 + nc]
        mask = x[:, 4 + nc : 4 + nc + nm]

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate(
                (box[i], x[i, 4 + j, None], j[:, None].astype(float), mask[i]),
                axis=1,
            )
        else:  # best class only
            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(float), mask), axis=1)[
                conf.flatten() > conf_thres
            ]
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[np.argsort(x[:, 4])[::-1][:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        if task == "obb":
            boxes = np.concatenate(
                (x[:, :2] + c, x[:, 2:4], x[:, -1:]), axis=-1
            )  # xywhr
            i = numpy_nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c
            # i = numpy_nms(boxes, scores, iou_thres)
        i = i[:max_det]
        # if merge and (1 < n < 3e3):
        #     iou = box_iou(boxes[i], boxes) > iou_thres
        #     weights = iou * scores[None]
        #     x[i, :4] = np.dot(weights, x[:, :4]) / weights.sum(
        #         1, keepdims=True
        #     )
        #     if redundant:
        #         i = i[iou.sum(1) > 1]

        output[xi] = x[i]

    return output


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


def box_label(
    im,
    box,
    label="",
    line_width=2,
    color=(128, 128, 128),
    txt_color=(255, 255, 255),
    rotated=False,
):
    """Add one xyxy box to image with label."""
    lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
    tf = max(lw - 1, 1)  # font thickness
    sf = lw / 3  # font scale
    if rotated:
        p1 = [int(b) for b in box[0]]
        # NOTE: cv2-version polylines needs np.asarray type.
        cv2.polylines(im, [np.asarray(box, dtype=int)], True, color, lw)
    else:
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[
            0
        ]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            im,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            sf,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.array): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


class OnnxBaseModel:
    def __init__(self, model_path, device_type: str = "gpu"):
        self.sess_opts = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(
                os.environ["OMP_NUM_THREADS"]
            )

        self.providers = ["CPUExecutionProvider"]
        if device_type.lower() == "gpu":
            self.providers = ["CUDAExecutionProvider"]

        self.ort_session = ort.InferenceSession(
            model_path,
            providers=self.providers,
            sess_options=self.sess_opts,
        )

    def get_ort_inference(
        self, blob, inputs=None, extract=True, squeeze=False
    ):
        if inputs is None:
            inputs = self.get_input_name()
            outs = self.ort_session.run(None, {inputs: blob})
        else:
            outs = self.ort_session.run(None, inputs)
        if extract:
            outs = outs[0]
        if squeeze:
            outs = outs.squeeze(axis=0)
        return outs

    def get_input_name(self):
        return self.ort_session.get_inputs()[0].name

    def get_input_shape(self):
        return self.ort_session.get_inputs()[0].shape

    def get_output_name(self):
        return [out.name for out in self.ort_session.get_outputs()]


class YOLOv8_OBB:
    """Oriented object detection model using yolov8_obb"""

    colors = Colors()

    def __init__(self, model_config=None) -> None:
        self.config = model_config
        model_abs_path = self.config["model_path"]
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(print("file not found: ", model_abs_path))

        self.net = OnnxBaseModel(
            model_abs_path, device_type=self.config["device"]
        )
        print("Successful loading model!")
        self.class_names = self.config["class_names"]
        self.nc = len(self.class_names)
        self.save_path = self.config["save_path"]
        self.iou_thres = self.config["iou_thres"]
        self.conf_thres = self.config["conf_thres"]
        self.input_shape = self.net.get_input_shape()[-2:]
        print(f"Input shape: {self.input_shape}")

    def preprocess(self, image):
        input_img = letterbox(image, self.input_shape)[0]
        # Transpose
        input_img = input_img[..., ::-1].transpose(2, 0, 1)
        # Expand
        input_img = input_img[np.newaxis, :, :, :].astype(np.float32)
        # Contiguous
        input_img = np.ascontiguousarray(input_img)
        # Norm
        blob = input_img / 255.0
        return blob

    def postprocess(self, prediction, image):
        p = non_max_suppression_v8(
            prediction,
            task="obb",
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            nc=self.nc,
        )
        img_shape = image.shape[:2]
        results = []
        for pred in p:
            pred[:, :4] = scale_boxes(
                self.input_shape, pred[:, :4], img_shape, xywh=True
            )
            # xywh, r, conf, cls
            results = np.concatenate(
                [pred[:, :4], pred[:, -1:], pred[:, 4:6]], axis=-1
            )
        return results

    def predict_shapes(self, image):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        blob = self.preprocess(image)
        start_time = time.time()
        outputs = self.net.get_ort_inference(blob)
        end_time = time.time()
        print("Inference time: {:.3f}s".format(end_time - start_time))
        results = self.postprocess(outputs, image)
        batch_xywhr = results[:, :5]
        batch_conf = results[:, -2]
        batch_clas = results[:, -1]
        for xywhr, conf, clas in zip(batch_xywhr, batch_conf, batch_clas):
            c = int(clas)
            label = self.class_names[c] + f": {conf:.2f}"
            xyxyxyxy = xywhr2xyxyxyxy(xywhr)
            box = xyxyxyxy.reshape(-1, 4, 2).squeeze()
            box_label(
                image, box, label, color=self.colors(c, True), rotated=True
            )
        if self.save_path:
            cv2.imwrite(self.save_path, image)


if __name__ == "__main__":
    save_path = ""
    image_path = "assets/examples/demo_obb.png"
    model_path = "path/to/yolov8/obb/yolov8s-obb.onnx"
    device = "cpu"
    iou_thres = 0.7
    conf_thres = 0.25
    class_names = [
        "plane",
        "ship",
        "storage tank",
        "baseball diamond",
        "tennis court",
        "basketball court",
        "ground track field",
        "harbor",
        "bridge",
        "large vehicle",
        "small vehicle",
        "helicopter",
        "roundabout",
        "soccer ball field",
        "swimming pool",
    ]
    configs = {
        "model_path": model_path,
        "device": device,
        "save_path": save_path,
        "iou_thres": iou_thres,
        "conf_thres": conf_thres,
        "class_names": class_names,
    }

    model = YOLOv8_OBB(configs)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    results = model.predict_shapes(image)
