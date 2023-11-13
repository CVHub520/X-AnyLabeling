import os
import re
import platform

from setuptools import find_packages, setup


def get_version():
    """Get package version from app_info.py file"""
    filename = "anylabeling/app_info.py"
    with open(filename, encoding="utf-8") as f:
        match = re.search(
            r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M
        )
    if not match:
        raise RuntimeError(f"{filename} doesn't contain __version__")
    version = match.groups()[0]
    return version


def get_preferred_device():
    """Get preferred device from app_info.py file: CPU or GPU"""
    filename = "anylabeling/app_info.py"
    with open(filename, encoding="utf-8") as f:
        match = re.search(
            r"""^__preferred_device__ = ['"]([^'"]*)['"]""", f.read(), re.M
        )
    if not match:
        raise RuntimeError(f"{filename} doesn't contain __preferred_device__")
    device = match.groups()[0]
    return device


def get_package_name():
    """Get package name based on context"""
    package_name = "x-anylabeling"
    preferred_device = get_preferred_device()
    if preferred_device == "GPU" and platform.system() != "Darwin":
        package_name = "x-anylabeling-gpu"
    return package_name


def get_install_requires():
    """Get python requirements based on context"""
    install_requires = [
        "imgviz>=0.11",
        "natsort>=7.1.0",
        "numpy",
        "Pillow>=2.8",
        "PyYAML",
        "termcolor",
        "opencv-python-headless",
        'PyQt5>=5.15.7; platform_system != "Darwin"',
        "onnx==1.13.1",
        "qimage2ndarray==1.10.0",
        "lap==0.4.0",
        "tqdm",
        "scipy",
        "shapely",
        "pyclipper",
        "filterpy",
        "tokenizers",
    ]

    # Add onnxruntime-gpu if GPU is preferred
    # otherwise, add onnxruntime.
    # Note: onnxruntime-gpu is not available on macOS
    preferred_device = get_preferred_device()
    if preferred_device == "GPU" and platform.system() != "Darwin":
        install_requires.append("onnxruntime-gpu==1.16.0")
        print("Building AnyLabeling with GPU support")
    else:
        install_requires.append("onnxruntime==1.16.0")
        print("Building AnyLabeling without GPU support")

    return install_requires


def get_long_description():
    """Read long description from README"""
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
    return long_description


setup(
    name=get_package_name(),
    version=get_version(),
    packages=find_packages(),
    description="Advanced Auto Labeling Solution with Added Features",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="CVHub",
    author_email="cvhub.cn@gmail.com",
    url="https://github.com/CVHub520/X-AnyLabeling",
    install_requires=get_install_requires(),
    license="GPLv3",
    keywords="Image Annotation, Machine Learning, Deep Learning",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "anylabeling=anylabeling.app:main",
        ],
    },
)
