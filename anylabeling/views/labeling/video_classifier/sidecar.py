import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .config import SCHEMA_VERSION, SIDECAR_TYPE
from .utils import ms_to_frame, safe_stem


@dataclass
class Segment:
    id: str
    label: str
    start_ms: int
    end_ms: int
    start_frame: int = 0
    end_frame: int = 0
    description: str = ""

    @classmethod
    def new(cls, label, start_ms, end_ms, fps=0.0, description=""):
        if end_ms < start_ms:
            start_ms, end_ms = end_ms, start_ms
        return cls(
            id=f"s{uuid.uuid4().hex[:10]}",
            label=label,
            start_ms=int(start_ms),
            end_ms=int(end_ms),
            start_frame=ms_to_frame(start_ms, fps),
            end_frame=ms_to_frame(end_ms, fps),
            description=description or "",
        )

    def duration_ms(self):
        return max(0, int(self.end_ms) - int(self.start_ms))

    def refresh_frames(self, fps):
        self.start_frame = ms_to_frame(self.start_ms, fps)
        self.end_frame = ms_to_frame(self.end_ms, fps)


@dataclass
class SidecarData:
    video: str = ""
    fps: float = 0.0
    duration_ms: int = 0
    width: int = 0
    height: int = 0
    labels: List[str] = field(default_factory=list)
    label_colors: Dict[str, str] = field(default_factory=dict)
    segments: List[Segment] = field(default_factory=list)

    def to_json(self):
        return {
            "version": SCHEMA_VERSION,
            "type": SIDECAR_TYPE,
            "video": self.video,
            "fps": float(self.fps or 0.0),
            "duration_ms": int(self.duration_ms or 0),
            "width": int(self.width or 0),
            "height": int(self.height or 0),
            "labels": list(self.labels),
            "label_colors": dict(self.label_colors),
            "segments": [asdict(s) for s in self.segments],
        }

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Sidecar payload must be a JSON object")
        sd = cls(
            video=str(data.get("video") or ""),
            fps=float(data.get("fps") or 0.0),
            duration_ms=int(data.get("duration_ms") or 0),
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
            labels=list(data.get("labels") or []),
            label_colors=dict(data.get("label_colors") or {}),
        )
        for entry in data.get("segments") or []:
            if not isinstance(entry, dict):
                continue
            try:
                seg = Segment(
                    id=str(entry.get("id") or f"s{uuid.uuid4().hex[:10]}"),
                    label=str(entry.get("label") or ""),
                    start_ms=int(entry.get("start_ms") or 0),
                    end_ms=int(entry.get("end_ms") or 0),
                    start_frame=int(entry.get("start_frame") or 0),
                    end_frame=int(entry.get("end_frame") or 0),
                    description=str(entry.get("description") or ""),
                )
            except (TypeError, ValueError):
                continue
            sd.segments.append(seg)
        return sd

    def upsert_label(self, label, color=None):
        if not label:
            return
        if label not in self.labels:
            self.labels.append(label)
        if color:
            self.label_colors[label] = color

    def remove_label(self, label, cascade=True):
        if label in self.labels:
            self.labels.remove(label)
        self.label_colors.pop(label, None)
        if cascade:
            self.segments = [s for s in self.segments if s.label != label]

    def rename_label(self, old, new):
        if not new or old == new:
            return
        if old in self.labels:
            idx = self.labels.index(old)
            if new in self.labels:
                self.labels.pop(idx)
            else:
                self.labels[idx] = new
        if old in self.label_colors:
            color = self.label_colors.pop(old)
            self.label_colors.setdefault(new, color)
        for seg in self.segments:
            if seg.label == old:
                seg.label = new


def sidecar_path_for(video_path):
    if not video_path:
        return ""
    stem = safe_stem(video_path)
    return os.path.join(
        os.path.dirname(os.path.abspath(video_path)), stem + ".json"
    )


def load_sidecar(video_path) -> Optional[SidecarData]:
    sp = sidecar_path_for(video_path)
    if not sp or not os.path.exists(sp):
        return None
    try:
        with open(sp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("type") and data.get("type") != SIDECAR_TYPE:
        # An unrelated sidecar (e.g. image annotation) — ignore.
        return None
    try:
        sd = SidecarData.from_json(data)
    except Exception:
        return None
    return sd


def save_sidecar(video_path, sidecar: SidecarData) -> str:
    sp = sidecar_path_for(video_path)
    if not sp:
        raise ValueError("Cannot resolve sidecar path: empty video_path")
    sidecar.video = os.path.basename(video_path)
    payload = sidecar.to_json()
    directory = os.path.dirname(sp) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".xva_", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, sp)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return sp
