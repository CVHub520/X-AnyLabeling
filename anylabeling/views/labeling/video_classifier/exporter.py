import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .sidecar import load_sidecar
from .utils import detect_ffmpeg, ms_to_seconds, safe_stem

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\-]+")


def safe_token(name, fallback="x"):
    s = _SAFE_NAME_RE.sub("_", (name or "").strip())
    s = s.strip("._")
    return s or fallback


class ExportConfig:
    def __init__(
        self,
        source_folder,
        output_dir,
        include_video=True,
        include_rawframes=False,
        re_encode=False,
        rawframe_fps=0,
        zip_output=True,
        only_video_path=None,
        only_segment_id=None,
    ):
        self.source_folder = source_folder
        self.output_dir = output_dir
        self.include_video = bool(include_video)
        self.include_rawframes = bool(include_rawframes)
        self.re_encode = bool(re_encode)
        self.rawframe_fps = int(rawframe_fps or 0)  # 0 = use video native fps
        self.zip_output = bool(zip_output)
        self.only_video_path = only_video_path  # if set, restrict to this file
        self.only_segment_id = only_segment_id


class ExporterWorker(QObject):
    progressChanged = pyqtSignal(int, str)  # 0..100, message
    finished = pyqtSignal(bool, str)  # ok, message
    logged = pyqtSignal(str)

    def __init__(self, config: ExportConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def is_cancelled(self):
        return self._cancel.is_set()

    # ----------------------------------------------------------------
    def run(self):
        cfg = self.config
        try:
            ffmpeg = detect_ffmpeg()
            if not ffmpeg:
                self.finished.emit(
                    False,
                    "ffmpeg was not found. Install ffmpeg or "
                    "imageio-ffmpeg to export clips.",
                )
                return

            pairs = self._scan_pairs(cfg)
            if not pairs:
                self.finished.emit(False, "No video+sidecar pairs found")
                return

            tasks, label_map = self._collect_tasks(pairs)
            if not tasks:
                self.finished.emit(False, "No segments to export")
                return
            self.logged.emit(
                f"Collected {len(tasks)} segments across {len(label_map)} labels"
            )

            os.makedirs(cfg.output_dir, exist_ok=True)

            counters = {}

            total = len(tasks)
            done = 0

            for task in tasks:
                if self.is_cancelled():
                    self.finished.emit(False, "Cancelled")
                    return
                video_token = safe_token(task["video_stem"])
                label_token = safe_token(task["label"])
                base_name = f"{video_token}_{task['seg_id']}"

                if cfg.include_video:
                    rel = os.path.join(
                        "videos", label_token, base_name + ".mp4"
                    )
                    out_path = os.path.join(cfg.output_dir, rel)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    ok = self._cut_video(ffmpeg, task, out_path, cfg.re_encode)
                    if not ok and not cfg.re_encode:
                        self.logged.emit(
                            "[warn] stream copy failed; retrying re-encode"
                        )
                        ok = self._cut_video(ffmpeg, task, out_path, True)
                    if not ok:
                        self.logged.emit(f"[warn] video cut failed: {rel}")

                if cfg.include_rawframes:
                    rel_dir = os.path.join("rawframes", label_token, base_name)
                    out_dir = os.path.join(cfg.output_dir, rel_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    n_frames = self._cut_frames(ffmpeg, task, out_dir, cfg)
                    if n_frames <= 0:
                        self.logged.emit(f"[warn] rawframes failed: {rel_dir}")

                counters[task["label"]] = counters.get(task["label"], 0) + 1
                done += 1
                self.progressChanged.emit(
                    int(done * 100 / max(1, total)),
                    f"{task['label']} {base_name}",
                )

            self._write_label_map(cfg.output_dir, label_map)

            meta = {
                "video": (
                    os.path.basename(cfg.only_video_path)
                    if cfg.only_video_path
                    else ""
                ),
                "include_video": cfg.include_video,
                "include_rawframes": cfg.include_rawframes,
                "re_encode": cfg.re_encode,
                "counts": counters,
                "num_classes": len(label_map),
                "label_map": list(label_map.keys()),
                "exported_at": int(time.time()),
            }
            with open(
                os.path.join(cfg.output_dir, "metadata.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            zip_path = ""
            if cfg.zip_output:
                if self.is_cancelled():
                    self.finished.emit(False, "Cancelled")
                    return
                self.progressChanged.emit(99, "Zipping output…")
                zip_path = self._zip_output(cfg.output_dir)

            tail = f" ({zip_path})" if zip_path else ""
            self.finished.emit(
                True,
                f"Exported {done} segments to {cfg.output_dir}{tail}",
            )
        except Exception as exc:
            self.finished.emit(False, f"Export failed: {exc}")

    # ----------------------------------------------------------------
    def _scan_pairs(self, cfg):
        if not cfg.only_video_path:
            return []
        video_paths = [cfg.only_video_path]

        pairs = []
        for vp in video_paths:
            sd = load_sidecar(vp)
            if sd is None:
                continue
            if not sd.segments:
                continue
            pairs.append((vp, sd))
        return pairs

    def _collect_tasks(self, pairs):
        tasks = []
        label_set = set()
        only_segment_id = self.config.only_segment_id
        for vp, sd in pairs:
            fps = float(sd.fps or 0.0)
            stem = safe_stem(vp)
            for seg in sd.segments:
                if only_segment_id and seg.id != only_segment_id:
                    continue
                if not seg.label:
                    continue
                if seg.end_ms <= seg.start_ms:
                    continue
                tasks.append(
                    {
                        "video_path": vp,
                        "video_stem": stem,
                        "fps": fps,
                        "seg_id": seg.id or f"s{uuid.uuid4().hex[:6]}",
                        "label": seg.label,
                        "start_ms": int(seg.start_ms),
                        "end_ms": int(seg.end_ms),
                    }
                )
                label_set.add(seg.label)
        labels_sorted = sorted(label_set)
        label_map = {name: idx for idx, name in enumerate(labels_sorted)}
        return tasks, label_map

    def _cut_video(self, ffmpeg, task, out_path, re_encode):
        s = ms_to_seconds(task["start_ms"])
        e = ms_to_seconds(task["end_ms"])
        duration = e - s
        if duration < 0.01:
            return False
        if re_encode:
            cmd = [
                ffmpeg,
                "-y",
                "-ss",
                f"{s:.3f}",
                "-i",
                task["video_path"],
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-an",
                out_path,
            ]
        else:
            cmd = [
                ffmpeg,
                "-y",
                "-ss",
                f"{s:.3f}",
                "-i",
                task["video_path"],
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-an",
                "-avoid_negative_ts",
                "make_zero",
                out_path,
            ]
        return self._run_ffmpeg(cmd)

    def _cut_frames(self, ffmpeg, task, out_dir, cfg):
        s = ms_to_seconds(task["start_ms"])
        e = ms_to_seconds(task["end_ms"])
        duration = e - s
        if duration < 0.01:
            return 0
        fps = cfg.rawframe_fps or task["fps"]
        if not fps or fps <= 0:
            fps = 25
        pattern = os.path.join(out_dir, "img_%05d.jpg")
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{s:.3f}",
            "-i",
            task["video_path"],
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            "-start_number",
            "1",
            pattern,
        ]
        if not self._run_ffmpeg(cmd):
            return 0
        try:
            return len([n for n in os.listdir(out_dir) if n.endswith(".jpg")])
        except OSError:
            return 0

    def _run_ffmpeg(self, cmd):
        try:
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NO_WINDOW", 0
                )
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                **kwargs,
            )
            stderr = b""
            while True:
                if self.is_cancelled():
                    proc.terminate()
                    try:
                        proc.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.communicate()
                    return False
                try:
                    _, stderr = proc.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if proc.returncode != 0:
                err = (
                    (stderr or b"").decode("utf-8", errors="replace")
                    if stderr
                    else ""
                )
                self.logged.emit(
                    f"[ffmpeg rc={proc.returncode}] {err.strip()[-400:]}"
                )
                return False
            return True
        except Exception as exc:
            self.logged.emit(f"[ffmpeg error] {exc}")
            return False

    def _write_label_map(self, root, label_map):
        with open(
            os.path.join(root, "label_map.txt"),
            "w",
            encoding="utf-8",
            newline="\n",
        ) as f:
            for label in label_map:  # dict insertion order = sorted
                f.write(label + "\n")

    def _zip_output(self, root):
        zip_path = os.path.join(
            os.path.dirname(os.path.abspath(root)),
            os.path.basename(os.path.abspath(root)) + ".zip",
        )
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for cur, _, files in os.walk(root):
                for name in files:
                    full = os.path.join(cur, name)
                    rel = os.path.relpath(full, os.path.dirname(root))
                    zf.write(full, rel)
        return zip_path


class ExporterController(QObject):
    """Runs the worker on a QThread; relays signals."""

    progressChanged = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    logged = pyqtSignal(str)

    def __init__(self, config: ExportConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._thread = QThread()
        self._worker = ExporterWorker(config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progressChanged.connect(self.progressChanged.emit)
        self._worker.logged.connect(self.logged.emit)
        self._worker.finished.connect(self._on_finished)

    def start(self):
        self._thread.start()

    def cancel(self):
        self._worker.cancel()

    def _on_finished(self, ok, msg):
        self.finished.emit(ok, msg)
        self._thread.quit()
        self._thread.wait(2000)
