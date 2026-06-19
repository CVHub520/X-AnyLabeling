from collections import deque

import numpy as np
import scipy.linalg

from .basetrack import TrackState
from .bot_sort import BOTrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH

_CORNER_DX_IDX = np.array([0, 0, 2, 2])
_CORNER_DY_IDX = np.array([1, 3, 1, 3])


def _nsa_kalman_update(kf, mean, covariance, measurement, confidence):
    w = max(1.0 - float(confidence), 0.05)
    std = kf._std_weight_position * mean[2:4]
    r = np.diag(np.square(np.r_[std, std])) * w
    h = kf._update_mat
    projected_mean = h @ mean
    projected_cov = np.linalg.multi_dot((h, covariance, h.T)) + r

    chol, lower = scipy.linalg.cho_factor(
        projected_cov, lower=True, check_finite=False
    )
    gain = scipy.linalg.cho_solve(
        (chol, lower), np.dot(covariance, h.T).T, check_finite=False
    ).T
    innovation = measurement - projected_mean
    new_mean = mean + innovation @ gain.T
    new_cov = covariance - np.linalg.multi_dot((gain, projected_cov, gain.T))
    return new_mean, new_cov


def _hmiou_distance(tracks_a, tracks_b):
    n, m = len(tracks_a), len(tracks_b)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32), np.ones(
            (n, m), dtype=np.float32
        )
    boxes_a = np.ascontiguousarray(
        [track.xyxy for track in tracks_a], dtype=np.float32
    )
    boxes_b = np.ascontiguousarray(
        [track.xyxy for track in tracks_b], dtype=np.float32
    )
    iou_sim = matching.bbox_ioa(boxes_a, boxes_b, iou=True)
    h_over = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T) - np.maximum(
        boxes_a[:, 1:2], boxes_b[:, 1:2].T
    )
    h_union = np.maximum(boxes_a[:, 3:4], boxes_b[:, 3:4].T) - np.minimum(
        boxes_a[:, 1:2], boxes_b[:, 1:2].T
    )
    h_iou = np.clip(h_over / (h_union + 1e-9), 0, 1)
    return iou_sim, 1.0 - h_iou * iou_sim


def _angle_distance(tracks, dets, frame_id, delta_t=3):
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float32)
    track_boxes = np.stack(
        [track.get_history_box(frame_id, delta_t) for track in tracks]
    )
    det_boxes = np.stack([det.xyxy for det in dets])
    deltas = det_boxes[None] - track_boxes[:, None]
    dx = deltas[:, :, _CORNER_DX_IDX]
    dy = deltas[:, :, _CORNER_DY_IDX]
    norms = np.sqrt(dx * dx + dy * dy) + 1e-5
    dx /= norms
    dy /= norms
    track_velocities = np.stack([track.velocity for track in tracks])
    dot = track_velocities[:, None, :, 0] * dx + (
        track_velocities[:, None, :, 1] * dy
    )
    dist = np.abs(np.arccos(np.clip(dot, -1, 1))).mean(axis=-1) / np.pi
    return dist * np.array([det.score for det in dets])[None]


def _confidence_distance(tracks, dets):
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float32)
    track_prev_scores = np.array([track.prev_score for track in tracks])
    track_curr_scores = np.array([track.score for track in tracks])
    track_proj_scores = track_curr_scores + (
        track_curr_scores - track_prev_scores
    )
    det_scores = np.array([det.score for det in dets])
    return np.abs(track_proj_scores[:, None] - det_scores[None])


def _iterative_associate(cost, match_thr, reduce_step=0.05):
    matches = []
    cost = cost.copy()
    while cost.shape[0] > 0 and cost.shape[1] > 0:
        nearest_det = np.argmin(cost, axis=1)
        nearest_track = np.argmin(cost, axis=0)
        new_matches = [
            [track_idx, nearest_det[track_idx]]
            for track_idx in range(cost.shape[0])
            if nearest_track[nearest_det[track_idx]] == track_idx
            and cost[track_idx, nearest_det[track_idx]] < match_thr
        ]
        if not new_matches:
            break
        matches.extend(new_matches)
        for track_idx, det_idx in new_matches:
            cost[track_idx, :] = np.inf
            cost[:, det_idx] = np.inf
        match_thr -= reduce_step
    matched_tracks = {track_idx for track_idx, _ in matches}
    matched_dets = {det_idx for _, det_idx in matches}
    unmatched_tracks = [
        i for i in range(cost.shape[0]) if i not in matched_tracks
    ]
    unmatched_dets = [i for i in range(cost.shape[1]) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def _track_aware_nms(tracks, dets, tai_thr, new_track_thresh):
    if not dets:
        return []
    scores = np.array([det.score for det in dets])
    allow = scores > new_track_thresh
    n_tracks = len(tracks)
    if n_tracks + len(dets) < 2:
        return allow.tolist()
    boxes = np.ascontiguousarray(
        [obj.xyxy for obj in tracks + dets], dtype=np.float32
    )
    iou = matching.bbox_ioa(boxes, boxes, iou=True)

    if n_tracks:
        allow &= iou[n_tracks:, :n_tracks].max(axis=1) <= tai_thr

    det_iou = iou[n_tracks:, n_tracks:]
    order = scores.argsort()[::-1]
    for i in order:
        if not allow[i]:
            continue
        suppress = det_iou[i] > tai_thr
        suppress[i] = False
        allow[suppress] = False
    return allow.tolist()


class TTSTrack(BOTrack):
    min_track_len = 3
    _delta_t = 3

    def __init__(self, xywh, score, cls):
        super().__init__(xywh, score, cls)
        self.prev_score = score
        self.velocity = np.zeros((4, 2), dtype=np.float32)
        self._history = deque(maxlen=self._delta_t + 1)

    def get_history_box(self, frame_id, dt):
        target = frame_id - dt
        for fid, box in self._history:
            if fid == target:
                return box.copy()
        if self._history:
            return self._history[-1][1].copy()
        return self.xyxy.copy()

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(
            self.convert_coords(self._tlwh)
        )
        self._history.append((frame_id, self.xyxy.copy()))
        self.tracklet_len = 0
        self.state = TrackState.New
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.prev_score = self.score
        self.mean, self.covariance = _nsa_kalman_update(
            self.kalman_filter,
            self.mean,
            self.covariance,
            self.convert_coords(new_track.tlwh),
            new_track.score,
        )
        self._history.append((frame_id, self.xyxy.copy()))
        self.score = new_track.score
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.cls, self.angle, self.idx = (
            new_track.cls,
            new_track.angle,
            new_track.idx,
        )

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.prev_score = self.score
        self.mean, self.covariance = _nsa_kalman_update(
            self.kalman_filter,
            self.mean,
            self.covariance,
            self.convert_coords(new_track.tlwh),
            new_track.score,
        )
        self._history.append((frame_id, new_track.xyxy.copy()))

        velocity = np.zeros((4, 2), dtype=np.float32)
        curr_box = new_track.xyxy
        for dt in range(1, self._delta_t + 1):
            delta = curr_box - self.get_history_box(frame_id, dt)
            dx, dy = delta[_CORNER_DX_IDX], delta[_CORNER_DY_IDX]
            norm = np.sqrt(dx * dx + dy * dy) + 1e-5
            velocity += np.stack([dx / norm, dy / norm], axis=-1) / dt
        self.velocity = velocity / self._delta_t

        self.score = new_track.score
        if self.state == TrackState.Tracked or (
            self.tracklet_len >= self.min_track_len
        ):
            self.state = TrackState.Tracked
            self.is_activated = True
        self.cls, self.angle, self.idx = (
            new_track.cls,
            new_track.angle,
            new_track.idx,
        )


class TRACKTRACK:
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = KalmanFilterXYWH()

        self.match_thr = getattr(args, "match_thresh", 0.7)
        self.lost_match_thr = getattr(args, "lost_match_thr", 0.0)
        self.penalty_p = getattr(args, "penalty_p", 0.2)
        self.reduce_step = getattr(args, "reduce_step", 0.05)
        self.conf_weight = getattr(args, "conf_weight", 0.1)
        self.angle_weight = getattr(args, "angle_weight", 0.05)
        self.tai_thr = getattr(args, "tai_thr", 0.55)
        self.new_track_thresh = getattr(args, "new_track_thresh", 0.7)
        self.min_track_len = getattr(args, "min_track_len", 3)
        self.gmc = GMC(method=getattr(args, "gmc_method", "sparseOptFlow"))
        self.reset_id()

    def update(self, scores, bboxes, cls, img=None):
        self.frame_id += 1
        activated, refind, lost, removed = [], [], [], []

        bboxes = np.concatenate(
            [bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1
        )
        high_mask = scores >= self.args.track_high_thresh
        low_mask = (scores > self.args.track_low_thresh) & (
            scores < self.args.track_high_thresh
        )

        dets_high = self.init_track(
            bboxes[high_mask], scores[high_mask], cls[high_mask]
        )
        dets_low = self.init_track(
            bboxes[low_mask], scores[low_mask], cls[low_mask]
        )

        unconfirmed, tracked = [], []
        for track in self.tracked_stracks:
            (unconfirmed if not track.is_activated else tracked).append(track)
        pool = self.joint_stracks(tracked, self.lost_stracks)

        if img is not None:
            warp = self.gmc.apply(img, [det.xyxy for det in dets_high])
            BOTrack.multi_gmc(pool, warp)
            BOTrack.multi_gmc(unconfirmed, warp)
        TTSTrack.multi_predict(pool)

        all_dets = dets_high + dets_low
        n_high = len(dets_high)
        cost = self.get_dists(pool, all_dets)
        if cost.shape[1] > n_high:
            cost[:, n_high:] += self.penalty_p
        cost = np.clip(cost, 0, 1)

        matches, unmatched_tracks, unmatched_dets = _iterative_associate(
            cost, self.match_thr, self.reduce_step
        )
        for track_idx, det_idx in matches:
            track, det = pool[track_idx], all_dets[det_idx]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)
        for track_idx in unmatched_tracks:
            track = pool[track_idx]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        leftover = [all_dets[i] for i in unmatched_dets if i < n_high]
        if unconfirmed and leftover:
            uc_cost = self.get_dists(unconfirmed, leftover)
            uc_matches, uc_unmatched_tracks, uc_unmatched_dets = (
                _iterative_associate(uc_cost, self.match_thr, self.reduce_step)
            )
            for track_idx, det_idx in uc_matches:
                unconfirmed[track_idx].update(leftover[det_idx], self.frame_id)
                activated.append(unconfirmed[track_idx])
            for track_idx in uc_unmatched_tracks:
                unconfirmed[track_idx].mark_removed()
                removed.append(unconfirmed[track_idx])
            leftover = [leftover[i] for i in uc_unmatched_dets]
        else:
            for track in unconfirmed:
                track.mark_removed()
                removed.append(track)

        if self.lost_match_thr > 0 and leftover:
            unmatched_lost = [
                t for t in pool if t.state == TrackState.Lost and t not in lost
            ]
            if unmatched_lost:
                lost_cost = self.get_dists(unmatched_lost, leftover)
                lost_matches, _, lost_unmatched = _iterative_associate(
                    lost_cost, self.lost_match_thr, self.reduce_step
                )
                for track_idx, det_idx in lost_matches:
                    unmatched_lost[track_idx].re_activate(
                        leftover[det_idx], self.frame_id, new_id=False
                    )
                    refind.append(unmatched_lost[track_idx])
                leftover = [leftover[i] for i in lost_unmatched]

        active = [
            track
            for track in self.tracked_stracks
            if track.state == TrackState.Tracked
        ] + activated
        for det, ok in zip(
            leftover,
            _track_aware_nms(
                active, leftover, self.tai_thr, self.new_track_thresh
            ),
        ):
            if ok:
                det.activate(self.kalman_filter, self.frame_id)
                activated.append(det)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = self.joint_stracks(
            self.tracked_stracks, activated
        )
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = self.sub_stracks(
            self.lost_stracks, self.tracked_stracks
        )
        self.lost_stracks.extend(lost)
        self.lost_stracks = self.sub_stracks(
            self.lost_stracks, self.removed_stracks
        )
        self.tracked_stracks, self.lost_stracks = (
            self.remove_duplicate_stracks(
                self.tracked_stracks, self.lost_stracks
            )
        )
        self.removed_stracks.extend(removed)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]

        return np.asarray(
            [
                x.result
                for x in self.tracked_stracks
                if x.is_activated and x.frame_id == self.frame_id
            ],
            dtype=np.float32,
        )

    def init_track(self, dets, scores, cls):
        tracks = []
        for xyxy, score, class_id in zip(dets, scores, cls):
            track = TTSTrack(xyxy, score, class_id)
            track.min_track_len = self.min_track_len
            tracks.append(track)
        return tracks

    def get_dists(self, tracks, detections):
        iou_sim, hmiou_dist = _hmiou_distance(tracks, detections)
        cost = hmiou_dist
        cost += self.conf_weight * _confidence_distance(tracks, detections)
        cost += self.angle_weight * _angle_distance(
            tracks, detections, self.frame_id
        )
        if iou_sim.size > 0:
            cost[iou_sim <= 0.10] = 1.0
        return np.clip(cost, 0, 1)

    @staticmethod
    def reset_id():
        TTSTrack.reset_id()

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYWH()
        self.reset_id()
        self.gmc.reset_params()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
