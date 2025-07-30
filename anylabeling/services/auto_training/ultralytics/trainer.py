import json
import os
import shutil
import threading
from datetime import datetime
from typing import List, Dict, Tuple, Callable

from .config import (
    DATASET_PATH,
    TASK_LABEL_MAPPINGS,
    TASK_SHAPE_MAPPINGS
)
from ._io import (
    load_yaml_config, 
    save_yaml_config,
)


def validate_basic_config(config: Dict) -> Tuple[bool, str]:
    basic = config.get("basic", {})

    if not basic.get("project", "").strip():
        return False, "Project field is required"

    if not basic.get("name", "").strip():
        return False, "Name field is required"

    model_path = basic.get("model", "").strip()
    if not model_path or not os.path.exists(model_path):
        return False, "Valid model file is required"

    data_path = basic.get("data", "").strip()
    if not data_path or not os.path.exists(data_path):
        return False, "Valid data file is required"

    return True, ""


def check_ultralytics_installation() -> Tuple[bool, str]:
    try:
        import ultralytics
        return True, ""
    except ImportError:
        return False, "Ultralytics is not installed. Please install it with: pip install ultralytics"


def create_yolo_dataset(image_list: List[str], task_type: str, dataset_ratio: float, data_file: str, output_dir: str = None, pose_cfg_file: str = None) -> str:
    from anylabeling.views.labeling.label_converter import LabelConverter

    data = load_yaml_config(data_file)
    if task_type == "pose":
        if not pose_cfg_file:
            return None, "Pose configuration file is required for pose detection tasks"
        converter = LabelConverter(pose_cfg_file=pose_cfg_file)
    else:
        converter = LabelConverter()
    converter.classes = [data['names'][i] for i in sorted(data['names'].keys())]

    data_file_name = os.path.splitext(os.path.basename(data_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.join(DATASET_PATH, task_type, f"{data_file_name.lower()}_{timestamp}")

    train_images_dir = os.path.join(temp_dir, "images", "train")
    val_images_dir = os.path.join(temp_dir, "images", "val")
    train_labels_dir = os.path.join(temp_dir, "labels", "train")
    val_labels_dir = os.path.join(temp_dir, "labels", "val")
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

    background_images = []
    mode = TASK_LABEL_MAPPINGS.get(task_type, "hbb")
    valid_images = []
    valid_shapes = TASK_SHAPE_MAPPINGS.get(task_type, [])

    for image_file in image_list:
        label_dir, filename = os.path.split(image_file)
        if output_dir:
            label_dir = output_dir
        label_file = os.path.join(label_dir, os.path.splitext(filename)[0] + ".json")

        if not os.path.exists(label_file):
            background_images.append(image_file)
            continue

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                label_info = json.load(f)
            shapes = label_info.get("shapes", [])
            has_valid_shape = any(
                shape.get("shape_type") in valid_shapes 
                for shape in shapes 
                if "shape_type" in shape
            )

            if has_valid_shape:
                valid_images.append((image_file, label_file))
            else:
                background_images.append(image_file)
        except Exception:
            background_images.append(image_file)
            continue

    train_count = int(len(valid_images) * dataset_ratio)
    train_valid_images = valid_images[:train_count]
    val_valid_images = valid_images[train_count:]
    all_train_images = [(img, None) for img in background_images] + train_valid_images

    for image_file, label_file in all_train_images:
        filename = os.path.basename(image_file)
        dst_image_path = os.path.join(train_images_dir, filename)

        if os.name == "nt":  # Windows
            shutil.copy2(image_file, dst_image_path)
        else:
            os.symlink(image_file, dst_image_path)

        if label_file and os.path.exists(label_file):
            dst_label_path = os.path.join(train_labels_dir, os.path.splitext(filename)[0] + ".txt")
            converter.custom_to_yolo(label_file, dst_label_path, mode, skip_empty_files=False)

    for image_file, label_file in val_valid_images:
        filename = os.path.basename(image_file)
        dst_image_path = os.path.join(val_images_dir, filename)

        if os.name == 'nt':
            shutil.copy2(image_file, dst_image_path)
        else:
            os.symlink(image_file, dst_image_path)

        if label_file and os.path.exists(label_file):
            dst_label_path = os.path.join(val_labels_dir, os.path.splitext(filename)[0] + ".txt")
            converter.custom_to_yolo(label_file, dst_label_path, mode, skip_empty_files=False)

    info_file = os.path.join(temp_dir, "dataset_info.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task type: {task_type}\n")
        f.write(f"Total images: {len(image_list)}\n")
        f.write(f"Train images: {len(all_train_images)}\n")
        f.write(f"Val images: {len(val_valid_images)}\n")
        f.write(f"Valid labeled images: {len(valid_images)}\n")
        f.write(f"Background images: {len(background_images)}\n")
        f.write(f"Dataset ratio: {dataset_ratio}\n")

    yaml_file = os.path.join(temp_dir, "data.yaml")
    data["path"] = temp_dir
    data["train"] = "images/train"
    data["val"] = "images/val"
    save_yaml_config(data, yaml_file)

    return temp_dir


class TrainingManager:
    def __init__(self):
        self.training_thread = None
        self.is_training = False
        self.callbacks = []

    def notify_callbacks(self, event_type: str, data: dict):
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception:
                pass

    def start_training(self, train_args: Dict) -> Tuple[bool, str]:
        if self.is_training:
            return False, "Training is already in progress"

        try:
            from ultralytics import YOLO
            model = YOLO(train_args.pop("model"))

            def train_in_thread():
                try:
                    self.is_training = True
                    self.notify_callbacks("training_started", {
                        "total_epochs": self.total_epochs
                    })
                    results = model.train(**train_args)

                    self.is_training = False
                    self.notify_callbacks("training_completed", {
                        "results": str(results),
                    })

                except Exception as e:
                    self.is_training = False
                    self.notify_callbacks("training_error", {
                        "error": str(e)
                    })

            self.training_thread = threading.Thread(target=train_in_thread)
            self.training_thread.daemon = True
            self.training_thread.start()

            return True, "Training started successfully"

        except ImportError:
            return False, "Ultralytics is not installed. Please install it with: pip install ultralytics"
        except Exception as e:
            return False, f"Failed to start training: {str(e)}"


_training_manager = TrainingManager()

def get_training_manager() -> TrainingManager:
    return _training_manager 