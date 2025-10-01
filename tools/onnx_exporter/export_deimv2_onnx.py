import argparse
import cv2
from PIL import Image

import numpy as np
import onnxruntime as ort

"""
[DEIMv2: Real-Time Object Detection Meets DINOv3](https://github.com/Intellindust-AI-Lab/DEIMv2) ONNX Inference Demo
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
Rewritten by Wei Wang (CVHub)

Export:
    python tools/deployment/export_onnx.py --check -c configs/deimv2/deimv2_hgnetv2_n_coco.yml -r deimv2_hgnetv2_n_coco.pth
    python tools/deployment/export_onnx.py --check -c configs/deimv2/deimv2_dinov3_s_coco.yml -r deimv2_dinov3_s_coco.pth
    python tools/deployment/export_onnx.py --check -c configs/deimv2/deimv2_dinov3_m_coco.yml -r deimv2_dinov3_m_coco.pth
    python tools/deployment/export_onnx.py --check -c configs/deimv2/deimv2_dinov3_l_coco.yml -r deimv2_dinov3_l_coco.pth
    python tools/deployment/export_onnx.py --check -c configs/deimv2/deimv2_dinov3_x_coco.yml -r deimv2_dinov3_x_coco.pth

Usage:
    python export_deimv2_onnx.py --model MODEL_PATH --source IMAGE_PATH [--conf CONF_THRES] [--imgsz SIZE]

Note:
    - Before exporting to ONNX, make sure to set batch size to 1.
    - For deimv2_hgnetv2_s_coco and deimv2_hgnetv2_m_coco, please download ViT-Tiny and ViT-Tiny+ weights first;
    - For deimv2_hgnetv2_l_coco and deimv2_hgnetv2_x_coco, please download dinov3-vits16 and dinov3-vits16plus weights first;
"""

COCO_CLASSES = (
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


def preprocess(image_path, img_size=640):
    """Preprocess image for model inference."""
    image = Image.open(image_path).convert("RGB")
    orig_size = (image.height, image.width)

    # Resize with aspect ratio preservation
    ratio = min(img_size / image.width, img_size / image.height)
    new_w, new_h = int(image.width * ratio), int(image.height * ratio)
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    # Pad to square
    padded = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    pad_w, pad_h = (img_size - new_w) // 2, (img_size - new_h) // 2
    padded.paste(resized, (pad_w, pad_h))

    # To tensor
    img_array = np.array(padded, dtype=np.float32).transpose(2, 0, 1)
    img_tensor = np.ascontiguousarray(img_array[None, :, :, :] / 255.0)

    return img_tensor, orig_size, ratio, (pad_w, pad_h)


def inference(model, img_tensor, img_size=640):
    """Run model inference."""
    orig_target_sizes = np.array([[img_size, img_size]], dtype=np.int64)
    outputs = model.run(
        None, {"images": img_tensor, "orig_target_sizes": orig_target_sizes}
    )
    return outputs


def postprocess(outputs, conf_thres, orig_size, ratio, padding):
    """Postprocess model outputs."""
    labels, boxes, scores = outputs

    # Filter by confidence
    mask = scores[0] >= conf_thres
    labels = labels[0][mask].astype(int)
    boxes = boxes[0][mask]
    scores = scores[0][mask]

    # Rescale boxes to original image size
    if len(boxes) > 0:
        pad_w, pad_h = padding
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio

        # Clip to image boundaries
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_size[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_size[0])
        boxes = boxes.astype(int)

    return labels, boxes, scores


def visualize(image_path, labels, boxes, scores, save_path=None):
    """Visualize detection results."""
    image = cv2.imread(image_path)

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=int)

    for label, box, score in zip(labels, boxes, scores):
        color = colors[label].tolist()
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        text = f"{COCO_CLASSES[label]}: {score:.2f}"
        cv2.putText(
            image,
            text,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    if save_path:
        cv2.imwrite(save_path, image)
        print(f"Result saved to: {save_path}")
    else:
        cv2.imshow("Detection Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main(args):
    """Main inference pipeline."""
    # Load model
    model = ort.InferenceSession(args.model)

    # Preprocess
    img_tensor, orig_size, ratio, padding = preprocess(args.source, args.imgsz)

    # Inference
    outputs = inference(model, img_tensor, args.imgsz)

    # Postprocess
    labels, boxes, scores = postprocess(
        outputs, args.conf, orig_size, ratio, padding
    )

    print(f"\nDetected {len(labels)} objects:")
    for i, (label, box, score) in enumerate(zip(labels, boxes, scores)):
        print(f"{i}: {COCO_CLASSES[label]} {box.tolist()} {score:.4f}")

    # Visualize
    save_path = (
        args.output
        if args.output.endswith((".jpg", ".png", ".jpeg"))
        else f"{args.output}.jpg"
    )
    visualize(args.source, labels, boxes, scores, save_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DEIMv2 ONNX Inference")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to ONNX model"
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size"
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="Confidence threshold"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="deimv2_result.jpg",
        help="Path to save output image",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
