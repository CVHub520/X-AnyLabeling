import numpy as np
from .tracker.ocsort import OCSort

__all__ = ['OcSort']

class OcSort:
    def __init__(self, input_shape: tuple, det_thresh: float = 0.2) -> None:

        self.tracker = OCSort(det_thresh=det_thresh)
        self.input_shape = input_shape

    def track(self, dets_xyxy: np.ndarray, image_shape: tuple) -> tuple:

        if isinstance(dets_xyxy, np.ndarray) and len(dets_xyxy) > 0:
            dets = self.tracker.update(dets_xyxy, image_shape)
            bbox_xyxy = dets[:, :4]
            ids = dets[:, 4]
            class_ids = dets[:, 5]
            scores = dets[:, 6]

            return bbox_xyxy, ids, scores, class_ids
        return [],[],[],[]
