import os
import cv2
import random
import numpy as np
import onnxruntime as ort

"""
The onnxruntime demo of the YOLOv10
Written by Wei Wang (CVHub)
    Usage:
        1. git clone https://github.com/THU-MIG/yolov10
        2. cd yolov10 and install the package
        3. export PYTHONPATH=/path/to/yolov10
        4. Download the corresponding weights
        5. Export the model to onnx format:
        ```bash
        yolo export model=/path/to/yolov10n/s/m/b/l/x.pt format=onnx opset=13 simplify
        ```
        7. Modified the paramters and run this script
        ```bash
        python ${export_yolov10_onnx.py}
        ```
"""


random.seed(10086)
CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
CLASS_COLORS = [
    [random.randint(0, 255) for _ in range(3)] for _ in range(len(CLASS_NAMES))
]


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


def rescale_coords(boxes, image_shape, input_shape):
    image_height, image_width = image_shape
    input_height, input_width = input_shape

    scale = min(input_width / image_width, input_height / image_height)

    pad_w = (input_width - image_width * scale) / 2
    pad_h = (input_height - image_height * scale) / 2

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height)

    return boxes.astype(int)


def preprocess(image, input_shape):
    # Resize
    input_img = letterbox(image, input_shape)[0]
    # Transpose
    input_img = input_img[..., ::-1].transpose(2, 0, 1)
    # Expand
    input_img = input_img[np.newaxis, :, :, :].astype(np.float32)
    # Contiguous
    input_img = np.ascontiguousarray(input_img)
    # Norm
    blob = input_img / 255.0
    return blob


def postprocess(outs, conf_thres, image_shape, input_shape):
    # Filtered by conf
    outs = outs[outs[:, 4] >= conf_thres]

    # Extract
    boxes = outs[:, :4]
    scores = outs[:, -2]
    labels = outs[:, -1].astype(int)

    # Rescale
    boxes = rescale_coords(boxes, image_shape, input_shape)

    return boxes, scores, labels


def main():
    conf_thres = 0.25
    input_shape = (640, 640)
    image_path = "/path/to/bus.jpg"
    save_path = ""
    model_path = "/path/to/yolov10n/s/m/b/l/x.onnx"

    ort_model = ort.InferenceSession(model_path)

    # Preprocess
    im0 = cv2.imread(image_path)
    image_shape = im0.shape[:2]
    blob = preprocess(im0, input_shape)

    # Inference
    outs = ort_model.run(None, {"images": blob})[0][0]

    # Postprocess
    boxes, scores, labels = postprocess(
        outs, conf_thres, image_shape, input_shape
    )

    # Draw the boxes
    for label, score, box in zip(labels, scores, boxes):
        label_text = f"{CLASS_NAMES[label]}: {score:.2f}"
        cv2.rectangle(
            im0, (box[0], box[1]), (box[2], box[3]), CLASS_COLORS[label], 2
        )
        cv2.putText(
            im0,
            label_text,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    try:
        cv2.imshow("image", im0)
        cv2.waitKey(0)
    except Exception as e:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, im0)
        print(f"Error displaying image: {e}")


if __name__ == "__main__":
    main()
