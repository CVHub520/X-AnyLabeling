from .tracker.byte_tracker import BYTETracker
import numpy as np
from ...utils.points_conversion import tlwh2xyxy


class ByteTrack(object):
    def __init__(self,
                 input_shape: tuple,
                 min_box_area: int = 10,
                 aspect_ratio_thresh: float= 3.0) -> None:

        self.min_box_area = min_box_area
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.input_shape = input_shape
        self.tracker = BYTETracker(frame_rate=30)

    def track(self,
              dets_xyxy: np.ndarray,
              image_shape: tuple) -> tuple:
        image_info = {"width": image_shape[0], "height": image_shape[1]}
        class_ids = []
        ids = []
        bboxes_xyxy = []
        scores = []
        if isinstance(dets_xyxy, np.ndarray) and len(dets_xyxy) > 0:
            class_ids = [int(i) for i in dets_xyxy[:, -1].tolist()]
            bboxes_xyxy, ids, scores = self._tracker_update(
                dets_xyxy,
                image_info,
            )
        return bboxes_xyxy, ids, scores, class_ids

    def _tracker_update(self, dets: np.ndarray, image_info: dict):
        online_targets = []
        if dets is not None:
            online_targets = self.tracker.update(
                dets[:, :-1],
                [image_info['height'], image_info['width']],
                [image_info['height'], image_info['width']],
            )
            
        online_xyxys = []
        online_ids = []
        online_scores = []
        for online_target in online_targets:
            tlwh = online_target.tlwh
            track_id = online_target.track_id
            vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                online_xyxys.append(tlwh2xyxy(tlwh))
                online_ids.append(track_id)
                online_scores.append(online_target.score)
        return online_xyxys, online_ids, online_scores
