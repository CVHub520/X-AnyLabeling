import cv2
import os
import random
import numpy as np
import onnxruntime as ort

from PIL import Image

"""
The onnxruntime demo of the D-FINE
Written by Wei Wang (CVHub)
    Usage:
        1. git clone https://github.com/Peterande/D-FINE.git
        2. cd D-FINE and install the package
        3. export PYTHONPATH=/path/to/D-FINE
        4. Download the corresponding weights
        5. Before you export, make sure change input shape as `data = torch.rand(1, 3, 640, 640)`, and then run the follow to onnx format:
        ```bash
        python tools/deployment/export_onnx.py --config configs/dfine/dfine_hgnetv2_s_coco.yml --resume weights/dfine_s_obj2coco.pth --check --simplify
        ```
        7. Modified the paramters and run this script
        ```bash
        python ${export_dfine_onnx.py}
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


def preprocess(input_image, input_size, interpolation=Image.BILINEAR):
    """
    Preprocesses the input image by resizing while preserving aspect ratio and adding padding.

    Args:
        input_image (PIL.Image): The input image to be processed.
        input_size (tuple): Input network size.
        interpolation: The interpolation method to use when resizing. Defaults to PIL.Image.BILINEAR.

    Returns:
        tuple: A tuple containing:
            - blob (np.ndarray): The preprocessed image as a normalized numpy array in NCHW format
            - orig_size (np.ndarray): Original image size as [[height, width]]
            - ratio (float): The resize ratio used
            - pad_w (int): The horizontal padding value
            - pad_h (int): The vertical padding value
    """
    image_w, image_h = input_image.size
    input_h, input_w = input_size

    ratio = min(input_w / image_w, input_h / image_h)
    new_width = int(image_w * ratio)
    new_height = int(image_h * ratio)

    image = input_image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (input_w, input_h))

    pad_h, pad_w = (input_h - new_height) // 2, (input_w - new_width) // 2
    new_image.paste(image, (pad_w, pad_h))

    orig_size = np.array(
        [[new_image.size[1], new_image.size[0]]], dtype=np.int64
    )
    im_data = np.array(new_image).astype(np.float32) / 255.0
    im_data = im_data.transpose(2, 0, 1)
    blob = np.expand_dims(im_data, axis=0)

    return blob, orig_size, ratio, pad_w, pad_h


def postprocess(outputs, orig_size, ratio, padding, conf_thres):
    """
    Post-processes the network's output.

    Args:
        outputs (list): The outputs from the network.
        orig_size (int, int): Original image size (img_w, img_h).
        ratio (float): The resize ratio.
        padding (tuple): Padding info (pad_w, pad_h).
        conf_thres (float): predict confidence

    Returns:
        labels, scores, boxes
    """
    labels, boxes, scores = outputs

    pad_w, pad_h = padding
    ori_w, ori_h = orig_size

    labels, scores, boxes = [], [], []

    # Only process boxes with scores above threshold
    for i, score in enumerate(scores[0]):
        if score > conf_thres:
            label_idx = int(labels[0][i])

            # Get box coordinates and adjust for padding and resize
            box = boxes[0][i]
            x1 = int((box[0] - pad_w) / ratio)
            y1 = int((box[1] - pad_h) / ratio)
            x2 = int((box[2] - pad_w) / ratio)
            y2 = int((box[3] - pad_h) / ratio)

            # Clip coordinates to image boundaries
            x1 = max(0, min(x1, ori_w))
            y1 = max(0, min(y1, ori_h))
            x2 = max(0, min(x2, ori_w))
            y2 = max(0, min(y2, ori_h))

            labels.append(label_idx)
            scores.append(score)
            boxes.append([x1, y1, x2, y2])

    return labels, scores, boxes


def main():
    conf_thres = 0.40
    input_shape = (640, 640)
    image_path = "/path/to/bus.jpg"
    save_path = ""
    model_path = "/path/to/dfine_s/m/l/x_obj2coco.onnx"

    ort_model = ort.InferenceSession(model_path)

    # Preprocess
    image = Image.open(image_path).convert("RGB")
    blob, orig_size, ratio, pad_w, pad_h = preprocess(image, input_shape)

    # Inference
    inputs = {"images": blob, "orig_target_sizes": orig_size}
    outputs = ort_model.run(None, inputs)

    # Postprocess
    labels, scores, boxes = postprocess(
        outputs, image.size, ratio, (pad_w, pad_h), conf_thres
    )

    # Draw the boxes
    im0 = cv2.imread(image_path)
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
