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


def rescale_box(input_shape, boxes, image_shape):
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
