import re
import platform
import subprocess
from importlib_metadata import version as get_package_version

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


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
    gpu_info = get_gpu_info()
    cuda_info = get_cuda_version()
    python_info = platform.python_version()
    pyqt5_info = get_installed_package_version('PyQt5')
    onnx_info = get_installed_package_version('onnx')
    ort_info = get_installed_package_version('onnxruntime')
    ort_gpu_info = get_installed_package_version('onnxruntime-gpu')
    opencv_contrib_info = get_installed_package_version('opencv-contrib-python-headless')

    system_info = {
        "Operating System": os_info,
        "CPU": cpu_info,
        "GPU": gpu_info,
        "CUDA": cuda_info,
        "Python Version": python_info,
    }
    pkg_info = {
        "PyQt5 Version": pyqt5_info,
        "ONNX Version": onnx_info,
        "ONNX Runtime Version": ort_info,
        "ONNX Runtime GPU Version": ort_gpu_info,
        "OpenCV Contrib Python Headless Version": opencv_contrib_info
    }

    return system_info, pkg_info


def get_installed_package_version(package_name):
    try:
        return get_package_version(package_name)
    except Exception:
        return None


def get_cuda_version():
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        version_line = next((line for line in nvcc_output.split('\n') if 'release' in line), None)
        if version_line:
            return version_line.split()[-1]
    except Exception:
        return None


def get_gpu_info():
    try:
        smi_output = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,name,memory.total', 
            '--format=csv,noheader,nounits'], encoding='utf-8')
        return ', '.join(smi_output.strip().split('\n'))
    except Exception:
        return None