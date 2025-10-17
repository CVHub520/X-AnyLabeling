import os
import re
import shutil
import math
import textwrap
import platform
import subprocess
import webbrowser
from difflib import SequenceMatcher
from importlib_metadata import version as get_package_version
from typing import Iterator, Tuple

try:
    import psutil
except ImportError:
    psutil = None


def format_bold(text):
    return f"\033[1m{text}\033[0m"


def format_color(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"


def gradient_text(
    text: str,
    start_color: Tuple[int, int, int] = (0, 0, 255),
    end_color: Tuple[int, int, int] = (255, 0, 255),
    frequency: float = 1.0,
) -> str:
    def color_function(t: float) -> Tuple[int, int, int]:
        def interpolate(start: float, end: float, t: float) -> float:
            # Use a sine wave for smooth, periodic interpolation
            return (
                start
                + (end - start) * (math.sin(math.pi * t * frequency) + 1) / 2
            )

        return tuple(
            round(interpolate(s, e, t)) for s, e in zip(start_color, end_color)
        )

    def gradient_gen(length: int) -> Iterator[Tuple[int, int, int]]:
        return (color_function(i / (length - 1)) for i in range(length))

    gradient = gradient_gen(len(text))
    return "".join(
        f"\033[38;2;{r};{g};{b}m{char}\033[0m"
        for char, (r, g, b) in zip(text, gradient)
    )  # noqa: E501


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def indent_text(text, indent=4):
    return textwrap.indent(text, " " * indent)


def is_chinese(s="人工智能"):
    # Is string composed of any Chinese characters?
    return bool(re.search("[\u4e00-\u9fff]", str(s)))


def is_possible_rectangle(points):
    if len(points) != 4:
        return False

    # Check if four points form a rectangle
    # The points are expected to be in the format:
    # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    dists = [square_dist(points[i], points[(i + 1) % 4]) for i in range(4)]
    dists.sort()

    # For a rectangle, the two smallest distances
    # should be equal and the two largest should be equal
    return dists[0] == dists[1] and dists[2] == dists[3]


def square_dist(p, q):
    # Calculate the square distance between two points
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def collect_system_info():
    os_info = platform.platform()
    cpu_info = platform.processor()
    cpu_count = os.cpu_count()

    if psutil:
        gib = 1 << 30
        ram = psutil.virtual_memory().total
        ram_info = f"{ram / gib:.1f} GB"
        total, used, free = shutil.disk_usage("/")
        disk_info = f"{(total - free) / gib:.1f}/{total / gib:.1f} GB"
    else:
        ram_info = "N/A (psutil not installed)"
        disk_info = "N/A (psutil not installed)"

    gpu_info = get_gpu_info()
    cuda_info = get_cuda_version()
    python_info = platform.python_version()
    pyqt5_info = get_installed_package_version("PyQt5")
    onnx_info = get_installed_package_version("onnx")
    ort_info = get_installed_package_version("onnxruntime")
    ort_gpu_info = get_installed_package_version("onnxruntime-gpu")
    opencv_contrib_info = get_installed_package_version(
        "opencv-contrib-python-headless"
    )

    system_info = {
        "Operating System": os_info,
        "CPU": cpu_info,
        "CPU Count": cpu_count,
        "RAM": ram_info,
        "Disk": disk_info,
        "GPU": gpu_info,
        "CUDA": cuda_info,
        "Python Version": python_info,
    }
    pkg_info = {
        "PyQt5 Version": pyqt5_info,
        "ONNX Version": onnx_info,
        "ONNX Runtime Version": ort_info,
        "ONNX Runtime GPU Version": ort_gpu_info,
        "OpenCV Contrib Python Headless Version": opencv_contrib_info,
    }

    return system_info, pkg_info


def find_most_similar_label(text, valid_labels):
    max_similarity = 0
    most_similar_label = valid_labels[0]

    for label in valid_labels:
        similarity = SequenceMatcher(None, text, label).ratio()
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_label = label

    return most_similar_label


def get_installed_package_version(package_name):
    try:
        return get_package_version(package_name)
    except Exception:
        return None


def get_cuda_version():
    try:
        nvcc_output = subprocess.check_output(["nvcc", "--version"]).decode(
            "utf-8"
        )
        version_line = next(
            (line for line in nvcc_output.split("\n") if "release" in line),
            None,
        )
        if version_line:
            return version_line.split()[-1]
    except Exception:
        return None


def get_gpu_info():
    try:
        smi_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        gpu_info_lines = []
        for line in smi_output.strip().split("\n"):
            parts = line.split(",")
            if len(parts) == 3:
                index = parts[0].strip()
                name = parts[1].strip()
                memory = parts[2].strip() + "MiB"
                gpu_info_lines.append(f"CUDA:{index} ({name}, {memory})")
        return ", ".join(gpu_info_lines)
    except Exception:
        return None


def open_url(url: str) -> None:
    """Open URL in browser while suppressing TTY warnings"""
    try:
        if platform.system() == "Linux":
            # Check if running in WSL
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower():
                    # Use powershell.exe for WSL
                    subprocess.run(
                        [
                            "powershell.exe",
                            "-Command",
                            f'Start-Process "{url}"',
                        ]
                    )
                else:
                    # For native Linux, use xdg-open
                    subprocess.run(
                        ["xdg-open", url], stderr=subprocess.DEVNULL
                    )
        else:
            webbrowser.open(url)
    except Exception:
        # Fallback to regular webbrowser.open
        webbrowser.open(url)
