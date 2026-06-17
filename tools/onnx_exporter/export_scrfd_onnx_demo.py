import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort

"""
The ONNXRuntime demo of SCRFD face detection with five-point landmarks.

Usage:
    python tools/onnx_exporter/export_scrfd_onnx_demo.py \
        --model /path/to/scrfd_10g_bnkps.onnx \
        --image /path/to/image.jpg \
        --output /path/to/output.jpg
"""


def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def preprocess(image, input_size):
    im_ratio = float(image.shape[0]) / image.shape[1]
    model_ratio = float(input_size[1]) / input_size[0]
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / image.shape[0]
    resized_img = cv2.resize(image, (new_width, new_height))
    det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized_img
    blob = cv2.dnn.blobFromImage(
        det_img,
        1.0 / 128,
        input_size,
        (127.5, 127.5, 127.5),
        swapRB=True,
    )
    return blob, det_scale


def get_anchor_centers(height, width, stride, num_anchors):
    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(
        np.float32
    )
    anchor_centers = (anchor_centers * stride).reshape((-1, 2))
    if num_anchors > 1:
        anchor_centers = np.stack(
            [anchor_centers] * num_anchors, axis=1
        ).reshape((-1, 2))
    return anchor_centers


def nms(dets, iou_threshold):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(overlap <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def detect(session, image, input_size, score_threshold, iou_threshold):
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.get_outputs()
    use_kps = len(outputs) == 9 or len(outputs) == 15
    fmc = 5 if len(outputs) in [10, 15] else 3
    feat_stride_fpn = [8, 16, 32, 64, 128] if fmc == 5 else [8, 16, 32]
    num_anchors = 1 if fmc == 5 else 2

    blob, det_scale = preprocess(image, input_size)
    net_outs = session.run(output_names, {input_name: blob})
    scores_list = []
    bboxes_list = []
    kpss_list = []

    for idx, stride in enumerate(feat_stride_fpn):
        scores = net_outs[idx][0].reshape(-1)
        bbox_preds = (net_outs[idx + fmc][0] * stride).reshape(-1, 4)
        height = input_size[1] // stride
        width = input_size[0] // stride
        anchor_centers = get_anchor_centers(height, width, stride, num_anchors)
        pos_inds = np.where(scores >= score_threshold)[0]
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        scores_list.append(scores[pos_inds, np.newaxis])
        bboxes_list.append(bboxes[pos_inds])
        if use_kps:
            kps_preds = (net_outs[idx + fmc * 2][0] * stride).reshape(-1, 10)
            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            kpss_list.append(kpss[pos_inds])

    if not scores_list or sum(len(scores) for scores in scores_list) == 0:
        return np.empty((0, 5), dtype=np.float32), None

    scores = np.vstack(scores_list)
    order = scores.ravel().argsort()[::-1]
    bboxes = np.vstack(bboxes_list) / det_scale
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms(pre_det, iou_threshold)
    det = pre_det[keep, :]
    if not use_kps:
        return det, None
    kpss = np.vstack(kpss_list) / det_scale
    kpss = kpss[order, :, :]
    kpss = kpss[keep, :, :]
    return det, kpss


def draw_results(image, det, kpss):
    for i, bbox in enumerate(det):
        x1, y1, x2, y2, score = bbox
        cv2.rectangle(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            image,
            f"{score:.2f}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
        if kpss is None:
            continue
        for point in kpss[i]:
            cv2.circle(
                image,
                (int(point[0]), int(point[1])),
                2,
                (0, 0, 255),
                2,
            )
    return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to SCRFD ONNX.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--output", default="scrfd_result.jpg", help="Path to output image."
    )
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.4)
    return parser.parse_args()


def main():
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    session = ort.InferenceSession(
        args.model, providers=["CPUExecutionProvider"]
    )
    input_size = (args.input_size, args.input_size)
    det, kpss = detect(
        session,
        image,
        input_size,
        args.score_threshold,
        args.iou_threshold,
    )
    result = draw_results(image, det, kpss)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(args.output, result)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
