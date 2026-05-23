import os
import re
import shutil
import subprocess

from PyQt6.QtGui import QColor, QImage

from anylabeling.views.labeling.utils.colormap import label_colormap

LABEL_COLORMAP = label_colormap()

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_SIZE_RE = re.compile(r"(?<![A-Za-z0-9])(\d{2,5})x(\d{2,5})(?![A-Za-z0-9])")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")


def ms_to_timecode(ms, with_ms=True):
    if ms is None or ms < 0:
        ms = 0
    total_seconds = int(ms // 1000)
    millis = int(ms % 1000)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        base = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        base = f"{minutes:02d}:{seconds:02d}"
    if with_ms:
        return f"{base}.{millis:03d}"
    return base


def ms_to_seconds(ms):
    return max(0.0, float(ms or 0) / 1000.0)


def color_from_index(index):
    if index < len(LABEL_COLORMAP) - 1:
        r, g, b = LABEL_COLORMAP[index + 1]
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    hue = (index * 47) % 360
    return QColor.fromHsv(hue, 175, 235).name()


def color_for_label(label, custom_colors=None, index=None):
    if custom_colors and label in custom_colors:
        return custom_colors[label]
    if index is not None:
        return color_from_index(max(0, int(index)))
    return color_from_index(0)


def detect_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def detect_ffprobe():
    path = shutil.which("ffprobe")
    if path:
        return path
    # imageio_ffmpeg only ships ffmpeg; fall back to ffmpeg -i parsing on caller.
    return None


def probe_video(video_path):
    """Return dict {fps, duration_ms, width, height} or None.

    Tries ffprobe first, ffmpeg stderr metadata second, then OpenCV.
    """
    info = _probe_with_ffprobe(video_path)
    if info:
        return info
    info = _probe_with_ffmpeg(video_path)
    if info:
        return info
    return _probe_with_opencv(video_path)


def _probe_with_ffprobe(video_path):
    ffprobe = detect_ffprobe()
    if not ffprobe or not os.path.exists(video_path):
        return None
    try:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1",
            video_path,
        ]
        out = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, timeout=15
        )
        text = out.decode("utf-8", errors="replace")
    except Exception:
        return None

    info = {"width": 0, "height": 0, "fps": 0.0, "duration_ms": 0}
    durations = []
    for key, value in _metadata_pairs(text):
        if key == "width":
            info["width"] = _safe_int(value)
        elif key == "height":
            info["height"] = _safe_int(value)
        elif key in ("r_frame_rate", "avg_frame_rate") and info["fps"] <= 0:
            info["fps"] = _parse_frame_rate(value)
        elif key == "duration":
            duration = _safe_float(value)
            if duration is not None:
                durations.append(duration)

    chosen = durations[-1] if durations else None
    if chosen and chosen > 0:
        info["duration_ms"] = int(round(chosen * 1000))
    if info["width"] and info["height"]:
        return info
    return None


def _metadata_pairs(text):
    for line in text.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        yield key.strip(), value.strip()


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_frame_rate(value):
    if not value or value in ("0/0", "N/A"):
        return 0.0
    if "/" in value:
        a, b = value.split("/", 1)
        try:
            a = float(a)
            b = float(b)
            if b <= 0:
                return 0.0
            return a / b
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _probe_with_ffmpeg(video_path):
    ffmpeg = detect_ffmpeg()
    if not ffmpeg or not os.path.exists(video_path):
        return None
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
        text = result.stdout.decode("utf-8", errors="replace")
    except Exception:
        return None

    info = {"width": 0, "height": 0, "fps": 0.0, "duration_ms": 0}
    match = _DURATION_RE.search(text)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        info["duration_ms"] = int(
            round(((hours * 60 + minutes) * 60 + seconds) * 1000)
        )

    video_line = ""
    for line in text.splitlines():
        if " Video:" in line or line.strip().startswith("Video:"):
            video_line = line
            break
    if video_line:
        match = _SIZE_RE.search(video_line)
        if match:
            info["width"] = int(match.group(1))
            info["height"] = int(match.group(2))
        match = _FPS_RE.search(video_line)
        if match:
            try:
                info["fps"] = float(match.group(1))
            except ValueError:
                pass

    if info["width"] and info["height"]:
        return info
    return None


def _probe_with_opencv(video_path):
    try:
        import cv2
    except ImportError:
        return None
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    duration_ms = 0
    if fps > 0 and frame_count > 0:
        duration_ms = int(round(frame_count / fps * 1000))
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration_ms": duration_ms,
    }


def ms_to_frame(ms, fps):
    if not fps or fps <= 0:
        return 0
    return int(round(float(ms) / 1000.0 * float(fps)))


def safe_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def is_video_path(path, exts):
    if not path:
        return False
    return path.lower().endswith(tuple(e.lower() for e in exts))


def extract_video_thumbnails(video_path, count=16, width=160):
    try:
        import cv2
    except ImportError:
        return []
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return []
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return []
        indices = _sample_frame_indices(frame_count, count)
        images = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            if w <= 0 or h <= 0:
                continue
            target_w = max(1, int(width))
            target_h = max(1, int(round(h * target_w / w)))
            rgb = cv2.resize(rgb, (target_w, target_h))
            qimage = QImage(
                rgb.data,
                target_w,
                target_h,
                rgb.strides[0],
                QImage.Format.Format_RGB888,
            )
            images.append(qimage.copy())
        return images
    finally:
        cap.release()


def _sample_frame_indices(frame_count, count):
    count = max(1, min(int(count or 1), frame_count))
    if count == 1:
        return [0]
    step = max(1, (frame_count - 1) / float(count - 1))
    return [min(frame_count - 1, int(round(i * step))) for i in range(count)]
