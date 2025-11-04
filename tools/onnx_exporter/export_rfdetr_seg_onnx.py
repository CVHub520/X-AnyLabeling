import cv2
import os
import random
import numpy as np
import onnxruntime as ort
from PIL import Image


"""
The onnxruntime demo of the Rf-DETR with instance segmentation
Written by Wei Wang (CVHub)
    Usage:
        1. https://github.com/roboflow/rf-detr.git
        2. cd rf-detr and install the package
        3. export PYTHONPATH=/path/to/rf-detr
        4. Download the corresponding weights
        5. Export the model to onnx format:
        ```bash
        from rfdetr import RFDETRSegPreview

        model = RFDETRSegPreview(pretrain_weights=<CHECKPOINT_PATH>)
        model.export()
        ```
        7. Modified the paramters and run this script
        ```bash
        python ${export_rfdetr_seg_onnx.py}
        ```
"""


random.seed(10086)
CLASS_NAMES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

CLASS_COLORS = {
    idx: [random.randint(0, 255) for _ in range(3)]
    for idx in CLASS_NAMES.keys()
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def box_cxcywh_to_xyxy(x):
    x_c = x[..., 0]
    y_c = x[..., 1]
    w = x[..., 2]
    h = x[..., 3]

    w = np.maximum(w, 0.0)
    h = np.maximum(h, 0.0)

    b = np.stack(
        [
            x_c - 0.5 * w,
            y_c - 0.5 * h,
            x_c + 0.5 * w,
            y_c + 0.5 * h,
        ],
        axis=-1,
    )

    return b


def preprocess(image, input_shape):
    if image.mode == "L":
        image = image.convert("RGB")

    image = image.resize(input_shape, Image.BILINEAR)

    image = np.array(image)

    image = image.astype(np.float32) / 255.0

    image = image.transpose((2, 0, 1))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(-1, 1, 1)
    image = (image - mean) / std

    image = np.expand_dims(image, axis=0)

    image = np.ascontiguousarray(image)

    return image


def postprocess(outs, conf_thres, num_select, image_shape):
    out_bbox = outs[0]
    out_logits = outs[1]
    out_masks = outs[2] if len(outs) == 3 else None

    prob = sigmoid(out_logits)
    prob_reshaped = prob.reshape(out_logits.shape[0], -1)

    topk_indexes = np.argpartition(-prob_reshaped, num_select, axis=1)[
        :, :num_select
    ]
    topk_values = np.take_along_axis(prob_reshaped, topk_indexes, axis=1)

    sort_indices = np.argsort(-topk_values, axis=1)
    topk_values = np.take_along_axis(topk_values, sort_indices, axis=1)
    topk_indexes = np.take_along_axis(topk_indexes, sort_indices, axis=1)

    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]

    boxes = box_cxcywh_to_xyxy(out_bbox)

    topk_boxes_expanded = np.expand_dims(topk_boxes, axis=-1)
    topk_boxes_tiled = np.tile(topk_boxes_expanded, (1, 1, 4))

    boxes = np.take_along_axis(boxes, topk_boxes_tiled, axis=1)
    img_h, img_w = image_shape
    scale_fct = np.array([[img_w, img_h, img_w, img_h]], dtype=np.float32)
    boxes = boxes * scale_fct[:, None, :]

    if out_masks is not None:
        masks = np.take_along_axis(
            out_masks, topk_boxes[:, :, None, None], axis=1
        )
        masks = masks[0]
        resized_masks = np.stack(
            [
                np.array(Image.fromarray(mask).resize((img_w, img_h)))
                for mask in masks
            ],
            axis=0,
        )
        masks = (resized_masks > 0).astype(np.uint8) * 255
    else:
        masks = None

    keep = scores[0] > conf_thres
    scores = scores[0][keep]
    labels = labels[0][keep]
    boxes = boxes[0][keep]
    if masks is not None:
        masks = masks[keep]

    return boxes, scores, labels, masks


def main():
    conf_thres = 0.5
    num_select = 300
    image_path = "/path/to/*.jpg"
    save_path = ""
    model_path = "/path/to/*.onnx"

    ort_model = ort.InferenceSession(model_path)
    input_info = ort_model.get_inputs()[0]
    input_shape = input_info.shape[2:]

    im0 = Image.open(image_path)
    image_shape = im0.size[::-1]
    blob = preprocess(im0, input_shape)

    outs = ort_model.run(None, {"input": blob})

    boxes, scores, labels, masks = postprocess(
        outs, conf_thres, num_select, image_shape
    )

    im0 = cv2.imread(image_path)
    if masks is not None:
        for i, mask in enumerate(masks):
            color = CLASS_COLORS[labels[i]]
            colored_mask = np.zeros_like(im0, dtype=np.uint8)
            colored_mask[mask > 0] = color
            im0 = cv2.addWeighted(im0, 1.0, colored_mask, 0.5, 0)

    for label, score, box in zip(labels, scores, boxes):
        box = box.astype(np.int32)
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

        print(f"Save the image to {save_path}")


if __name__ == "__main__":
    main()
