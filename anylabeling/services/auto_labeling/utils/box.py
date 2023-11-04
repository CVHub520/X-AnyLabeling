import numpy as np


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1, box2):
    area1 = box_area(box1)  # N
    area2 = box_area(box2)  # M
    # broadcasting
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou  # NxM


def rescale_box(input_shape, boxes, image_shape, kpts=False):
    '''Rescale the output to the original image shape'''
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
            boxes[:, i * 3]  = (boxes[:, i * 3] -  padding[1]) / ratio
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
    #lmdks 
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
    '''Rescale the output to the original image shape'''
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
    boxes[:, 2] = np.clip(
        (boxes[:, 0] + boxes[:, 2]), 0, image_shape[1]
    )  # x2
    boxes[:, 3] = np.clip(
        (boxes[:, 1] + boxes[:, 3]), 0, image_shape[0]
    )  # y2
    if kpts:
        num_kpts = boxes.shape[1] // 3
        for i in range(2, num_kpts + 1):
            boxes[:, i * 3 - 1] = (boxes[:, i * 3 - 1] - padding[0]) / ratio
            boxes[:, i * 3]  = (boxes[:, i * 3] -  padding[1]) / ratio
    return boxes


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
