# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import glob
import json
import ntpath
import os
import pickle
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np


def unzip(file_path: str, dest_dir: str):
    """
    Unzips compressed .zip file.
    Example inputs:
        file_path: 'data/01_alb_id.zip'
        dest_dir: 'data/'
    """

    # unzip file
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(dest_dir)


def save_json(data, save_path, indent: Optional[int] = None):
    """
    Saves json formatted data (given as "data") as save_path
    Example inputs:
        data: {"image_id": 5}
        save_path: "dirname/coco.json"
        indent: Train json files with indent=None, val json files with indent=4
    """
    # create dir if not present
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # export as json
    with open(save_path, "w", encoding="utf-8") as outfile:
        json.dump(
            data,
            outfile,
            separators=(",", ":"),
            cls=NumpyEncoder,
            indent=indent,
        )


# type check when save json files
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def load_json(load_path: str, encoding: str = "utf-8"):
    """
    Loads json formatted data (given as "data") from load_path
    Encoding type can be specified with 'encoding' argument

    Example inputs:
        load_path: "dirname/coco.json"
    """
    # read from path
    with open(load_path, encoding=encoding) as json_file:
        data = json.load(json_file)
    return data


def list_files(
    directory: str,
    contains: list = [".json"],
    verbose: int = 1,
) -> list:
    """
    Walk given directory and return a list of file path with desired extension

    Args:
        directory: str
            "data/coco/"
        contains: list
            A list of strings to check if the target file contains them, example: ["coco.png", ".jpg", "jpeg"]
        verbose: int
            0: no print
            1: print number of files

    Returns:
        filepath_list : list
            List of file paths
    """
    # define verboseprint
    verboseprint = print if verbose else lambda *a, **k: None

    filepath_list = []

    for file in os.listdir(directory):
        # check if filename contains any of the terms given in contains list
        if any(strtocheck in file.lower() for strtocheck in contains):
            filepath = os.path.join(directory, file)
            filepath_list.append(filepath)

    number_of_files = len(filepath_list)
    folder_name = Path(directory).name

    verboseprint(
        f"There are {str(number_of_files)} listed files in folder: {folder_name}/"
    )

    return filepath_list


def list_files_recursively(
    directory: str, contains: list = [".json"], verbose: str = True
) -> (list, list):
    """
    Walk given directory recursively and return a list of file path with desired extension

    Arguments
    -------
        directory : str
            "data/coco/"
        contains : list
            A list of strings to check if the target file contains them, example: ["coco.png", ".jpg", "jpeg"]
        verbose : bool
            If true, prints some results
    Returns
    -------
        relative_filepath_list : list
            List of file paths relative to given directory
        abs_filepath_list : list
            List of absolute file paths
    """

    # define verboseprint
    verboseprint = print if verbose else lambda *a, **k: None

    # walk directories recursively and find json files
    abs_filepath_list = []
    relative_filepath_list = []

    # r=root, d=directories, f=files
    for r, _, f in os.walk(directory):
        for file in f:
            # check if filename contains any of the terms given in contains list
            if any(strtocheck in file.lower() for strtocheck in contains):
                abs_filepath = os.path.join(r, file)
                abs_filepath_list.append(abs_filepath)
                relative_filepath = abs_filepath.split(directory)[-1]
                relative_filepath_list.append(relative_filepath)

    number_of_files = len(relative_filepath_list)
    folder_name = directory.split(os.sep)[-1]

    verboseprint(
        "There are {} listed files in folder {}.".format(
            number_of_files, folder_name
        )
    )

    return relative_filepath_list, abs_filepath_list


def get_base_filename(path: str):
    """
    Takes a file path, returns (base_filename_with_extension, base_filename_without_extension)
    """
    base_filename_with_extension = ntpath.basename(path)
    base_filename_without_extension, _ = os.path.splitext(
        base_filename_with_extension
    )
    return base_filename_with_extension, base_filename_without_extension


def get_file_extension(path: str):
    filename, file_extension = os.path.splitext(path)
    return file_extension


def load_pickle(load_path):
    """
    Loads pickle formatted data (given as "data") from load_path
    Example inputs:
        load_path: "dirname/coco.pickle"
    """
    # read from path
    with open(load_path) as json_file:
        data = pickle.load(json_file)
    return data


def save_pickle(data, save_path):
    """
    Saves pickle formatted data (given as "data") as save_path
    Example inputs:
        data: {"image_id": 5}
        save_path: "dirname/coco.pickle"
    """
    # create dir if not present
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # export as json
    with open(save_path, "wb") as outfile:
        pickle.dump(data, outfile)


def import_model_class(model_type, class_name):
    """
    Imports a predefined detection class by class name.

    Args:
        model_type: str
            "yolov5", "detectron2", "mmdet", "huggingface" etc
        model_name: str
            Name of the detection model class (example: "MmdetDetectionModel")
    Returns:
        class_: class with given path
    """
    module = __import__(f"sahi.models.{model_type}", fromlist=[class_name])
    class_ = getattr(module, class_name)
    return class_


def increment_path(path, exist_ok=True, sep=""):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def download_from_url(from_url: str, to_path: str):
    Path(to_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(to_path):
        urllib.request.urlretrieve(
            from_url,
            to_path,
        )


def is_colab():
    import sys

    # Is environment a Google Colab instance?
    return "google.colab" in sys.modules
