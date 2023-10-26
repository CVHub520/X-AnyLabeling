import numpy as np
from .__base__.yolo import YOLO

from .utils import (
    numpy_nms,
    xywh2xyxy,
)

class YOLOv8(YOLO):

    def postprocess(
            self, 
            prediction, 
            multi_label=False, 
            max_det=1000,
        ):
        prediction = prediction.transpose((0, 2, 1))
        num_classes = prediction.shape[2] - 4
        pred_candidates = np.max(prediction[..., 4:], axis=-1) > self.conf_thres

        max_wh = 4096
        max_nms = 30000
        multi_label &= num_classes > 1

        output = [np.zeros((0, 6))] * prediction.shape[0]
        for img_idx, x in enumerate(prediction):  # image index, image inference
            x = x[pred_candidates[img_idx]]  # confidence

            # If no box remains, skip the next process.
            if not x.shape[0]:
                continue

            # (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
            if multi_label:
                box_idx, class_idx = np.nonzero(x[:, 4:] > self.conf_thres)
                box = box[box_idx]
                conf = x[box_idx, class_idx + 4][:, None]
                class_idx = class_idx[:, None].astype(float)
                x = np.concatenate((box, conf, class_idx), axis=1)
            else:
                conf = np.max(x[:, 4:], axis=1, keepdims=True)
                class_idx = np.argmax(x[:, 4:], axis=1)
                x = np.concatenate(
                    (box, conf, class_idx[:, None].astype(float)), axis=1
                )[conf.flatten() > self.conf_thres]

            # Filter by class, only keep boxes whose category is in classes.
            if self.filter_classes:
                x = x[(x[:, 5:6] == np.array(self.filter_classes)).any(1)]

            # Check shape
            num_box = x.shape[0]  # number of boxes
            if not num_box:  # no boxes kept.
                continue
            elif num_box > max_nms:  # excess max boxes' number.
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            class_offset = x[:, 5:6] * (0 if self.agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
            keep_box_idx = numpy_nms(boxes, scores, self.nms_thres)  # NMS
            if keep_box_idx.shape[0] > max_det:  # limit detections
                keep_box_idx = keep_box_idx[:max_det]

            output[img_idx] = x[keep_box_idx]

        return output