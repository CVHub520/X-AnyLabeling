import os
import json
import shutil
import re
import base64
from collections import deque
from typing import List, Dict

from PyQt6.QtCore import QObject, Qt, QTemporaryFile
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QMessageBox, QProgressDialog

from anylabeling.views.labeling.chatbot.utils import save_json
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.schema import create_xlabel_template
from anylabeling.views.labeling.vqa.utils import AIWorkerThread

from anylabeling.views.labeling.utils.general import (
    resolve_export_directory,
    resolve_path_within_directory,
)


def get_label_file_path(image_path: str, output_dir: str = None) -> str:
    if output_dir:
        image_name = os.path.basename(image_path)
        label_name = os.path.splitext(image_name)[0] + ".json"
        return os.path.join(output_dir, label_name)
    else:
        return os.path.splitext(image_path)[0] + ".json"


def load_flags_from_json(json_path: str) -> Dict[str, bool]:
    if not os.path.exists(json_path):
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("flags", {})
    except (json.JSONDecodeError, KeyError):
        return {}


def save_flags_to_json(json_path: str, flags: Dict[str, bool]):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["flags"] = flags

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_first_true_flag(flags: Dict[str, bool]) -> str:
    for key, value in flags.items():
        if value:
            return key
    return None


def export_image_to_category(image_path: str, category: str, output_dir: str):
    category_dir = resolve_export_directory(output_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    image_name = os.path.basename(image_path)
    dest_path = resolve_path_within_directory(
        category_dir / image_name, output_dir
    )

    if not os.path.exists(dest_path):
        shutil.copy2(image_path, dest_path)


def get_display_text_for_flags(
    flags: Dict[str, bool], labels: List[str]
) -> str:
    if not flags:
        return ""

    active_labels = []
    for label in labels:
        if flags.get(label, False):
            active_labels.append(label)

    return ", ".join(active_labels) if active_labels else ""


def create_ai_prompt_template(labels: List[str], is_multiclass: bool) -> str:
    task_type = "multi-class" if is_multiclass else "multi-label"
    labels_str = ", ".join([f'"{label}"' for label in labels])

    if is_multiclass:
        instruction = "Set exactly ONE category to 'true' that best matches the image, keep all others as 'false'."
    else:
        instruction = "Set ALL applicable categories to 'true', keep non-applicable ones as 'false'."

    return f"""@image
You are an expert image classifier. Your task is to perform {task_type} classification.

Task Definition: Analyze the given image and classify it based on the provided categories.

Available Categories: [{labels_str}]

Instructions:
1. Carefully examine the image and identify the main subject and their activity
2. Be precise - only select categories that clearly match what you observe

Return your result in strict JSON format:
{{{", ".join([f'"{label}": false' for label in labels])}}}

{instruction}"""


class MultiTaskBatch(QObject):
    def __init__(self, owner, paths, prompt, concurrency):
        super().__init__(owner)
        self.owner = owner
        self.prompt = prompt
        self.concurrency = concurrency
        self.queue = deque()
        self.running = {}
        self.files = {}
        self.skipped = 0
        self.failed = 0
        self.succeeded = 0
        self.cancelled = False
        self.finished = False
        self.prepare(paths)
        self.total = len(self.queue)
        self.progress = QProgressDialog(
            self.tr("Processing objects..."),
            self.tr("Cancel"),
            0,
            self.total,
            owner,
        )
        self.progress.setWindowTitle(self.tr("Progress"))
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(False)
        self.progress.setAutoReset(False)
        self.progress.setMinimumWidth(440)
        self.progress.canceled.connect(self.cancel)

    def schema(self, shape):
        owner = self.owner.label_widget
        kinds = owner.attribute_widget_types.get(shape.label, {})
        return {
            name: "text" if kinds.get(name) == "lineedit" else list(options)
            for name, options in owner.attributes.get(shape.label, {}).items()
            if kinds.get(name) != "group_id"
        }

    def prepare(self, paths):
        owner = self.owner.label_widget
        for path in paths:
            if path == owner.filename:
                for shape in owner.canvas.shapes:
                    schema = self.schema(shape)
                    if schema:
                        self.queue.append((path, shape, schema, None))
                    else:
                        self.skipped += 1
                continue
            label_path = get_label_file_path(
                path, getattr(owner, "output_dir", None)
            )
            if not os.path.exists(label_path):
                self.skipped += 1
                continue
            try:
                with open(label_path, "rb") as stream:
                    original = stream.read()
                data = json.loads(original)
                self.files[label_path] = original
                for index, item in enumerate(data["shapes"]):
                    shape = Shape().load_from_dict(item)
                    schema = self.schema(shape)
                    if schema:
                        self.queue.append(
                            (path, shape, schema, (label_path, index, item))
                        )
                    else:
                        self.skipped += 1
            except (OSError, ValueError, KeyError, TypeError):
                self.failed += 1

    def start(self):
        if not self.total:
            QMessageBox.information(
                self.owner,
                self.tr("AI Assistant"),
                self.tr("No supported objects to process."),
            )
            self.deleteLater()
            return
        self.progress.show()
        self.advance()

    def advance(self):
        while (
            not self.cancelled
            and self.queue
            and len(self.running) < self.concurrency
        ):
            job = self.queue.popleft()
            path, shape, schema, target = job
            image = (
                self.owner.label_widget.image
                if target is None
                else QImage(path)
            )
            bounds = self.owner.image_label.object_path(shape).boundingRect()
            margin = max(bounds.width(), bounds.height(), 20) * 0.1
            rect = (
                bounds.adjusted(-margin, -margin, margin, margin)
                .toAlignedRect()
                .intersected(image.rect())
            )
            crop = QTemporaryFile()
            if image.isNull() or rect.isEmpty() or not crop.open():
                self.failed += 1
                continue
            crop_path = crop.fileName()
            crop.close()
            if not image.copy(rect).save(crop_path, "JPEG"):
                self.failed += 1
                continue
            prompt = self.prompt
            if "@image" not in prompt:
                prompt = "@image\n" + prompt
            if "@object" not in prompt:
                prompt += "\nObject category: @object"
            if "@attributes" not in prompt:
                prompt += "\nAttributes: @attributes"
            prompt = prompt.replace("@object", shape.label).replace(
                "@attributes", json.dumps(schema, ensure_ascii=False)
            )
            worker = AIWorkerThread(prompt, "", {}, crop_path, [], None)
            worker.crop_file = crop
            self.running[worker] = job
            self.owner._ai_workers.add(worker)
            worker.finished.connect(
                lambda result, success, error, w=worker: self.result(
                    w, result, success, error
                )
            )
            worker.completed.connect(lambda w=worker: self.completed(w))
            worker.start()
        self.update_progress()
        if not self.running and (not self.queue or self.cancelled):
            self.finish()

    def result(self, worker, result, success, error):
        if self.cancelled:
            return
        path, shape, schema, target = self.running[worker]
        try:
            if not success:
                raise ValueError(error)
            match = re.search(r"```json\s*(.*?)\s*```", result, re.DOTALL)
            values = json.loads(match.group(1) if match else result)
            if not isinstance(values, dict):
                raise ValueError("Invalid response")
            values = {
                name: value
                for name, value in values.items()
                if name in schema
                and isinstance(value, str)
                and value.strip()
                and (schema[name] == "text" or value in schema[name])
            }
            if not values:
                raise ValueError("No valid attributes")
            if target is None:
                owner = self.owner.label_widget
                if owner.filename != path or not any(
                    item is shape for item in owner.canvas.shapes
                ):
                    raise ValueError("Object no longer available")
                for name, value in values.items():
                    self.owner.multitask_form.set_value(shape, name, value)
                self.owner.select_object(
                    self.owner.object_combo.currentIndex() - 1
                )
            else:
                label_path, index, original_shape = target
                with open(label_path, "rb") as stream:
                    content = stream.read()
                if content != self.files[label_path]:
                    raise ValueError("Annotation changed during processing")
                data = json.loads(content)
                item = data["shapes"][index]
                if item.get("label") != original_shape.get(
                    "label"
                ) or item.get("points") != original_shape.get("points"):
                    raise ValueError("Object changed during processing")
                item.setdefault("attributes", {}).update(values)
                save_json(data, label_path)
                with open(label_path, "rb") as stream:
                    self.files[label_path] = stream.read()
            self.succeeded += 1
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            self.failed += 1
            self.owner._multitask_batch_errors.append(
                f"{os.path.basename(path)}: {exc}"
            )

    def completed(self, worker):
        if hasattr(worker, "crop_file"):
            worker.crop_file.remove()
        self.owner._ai_workers.discard(worker)
        self.running.pop(worker, None)
        self.advance()

    def update_progress(self):
        completed = self.total - len(self.queue) - len(self.running)
        self.progress.setLabelText(
            self.tr("Processed %d / %d objects") % (completed, self.total)
        )
        self.progress.setValue(completed)

    def cancel(self):
        self.cancelled = True
        self.queue.clear()
        for worker in self.running:
            worker.cancel()
        if not self.running:
            self.finish()

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.progress.canceled.disconnect(self.cancel)
        self.progress.close()
        if (
            not self.cancelled
            and self.succeeded
            and not getattr(self, "classification", False)
            and self.owner.label_widget.canvas.shapes
        ):
            self.owner.select_object(0)
            self.owner.image_label.focus_selected_object()
        message = self.tr("Completed: %d\nSkipped: %d\nFailed: %d") % (
            self.succeeded,
            self.skipped,
            self.failed,
        )
        if self.cancelled:
            message = self.tr("Processing cancelled.") + "\n\n" + message
        if self.owner._multitask_batch_errors:
            message += "\n\n" + "\n".join(
                self.owner._multitask_batch_errors[:5]
            )
        QMessageBox.information(self.owner, self.tr("AI Assistant"), message)
        self.owner._multitask_batch = None
        self.deleteLater()


class ClassificationBatch(MultiTaskBatch):
    def __init__(self, owner, paths, prompt, concurrency):
        self.classification = True
        self.labels = list(owner.labels)
        self.is_multiclass = owner.is_multiclass
        super().__init__(owner, paths, prompt, concurrency)
        self.progress.setLabelText(self.tr("Processing images..."))

    def prepare(self, paths):
        owner = self.owner.label_widget
        for path in paths:
            if path == owner.filename:
                self.queue.append((path, None))
                continue
            label_path = get_label_file_path(
                path, getattr(owner, "output_dir", None)
            )
            try:
                if os.path.exists(label_path):
                    with open(label_path, "rb") as stream:
                        self.files[label_path] = stream.read()
                else:
                    self.files[label_path] = None
                self.queue.append((path, label_path))
            except OSError as exc:
                self.failed += 1
                self.owner._multitask_batch_errors.append(
                    f"{os.path.basename(path)}: {exc}"
                )

    def advance(self):
        while (
            not self.cancelled
            and self.queue
            and len(self.running) < self.concurrency
        ):
            job = self.queue.popleft()
            path, _ = job
            prompt = (
                self.prompt
                if "@image" in self.prompt
                else "@image\n" + self.prompt
            )
            worker = AIWorkerThread(prompt, "", {}, path, [], None)
            self.running[worker] = job
            self.owner._ai_workers.add(worker)
            worker.finished.connect(
                lambda result, success, error, w=worker: self.result(
                    w, result, success, error
                )
            )
            worker.completed.connect(lambda w=worker: self.completed(w))
            worker.start()
        self.update_progress()
        if not self.running and (not self.queue or self.cancelled):
            self.finish()

    def result(self, worker, result, success, error):
        if self.cancelled:
            return
        path, label_path = self.running[worker]
        try:
            if not success:
                raise ValueError(error)
            match = re.search(r"```json\s*(.*?)\s*```", result, re.DOTALL)
            flags = json.loads(match.group(1) if match else result)
            if (
                not isinstance(flags, dict)
                or set(flags) != set(self.labels)
                or any(type(value) is not bool for value in flags.values())
            ):
                raise ValueError("Invalid classification result")
            if self.is_multiclass and sum(flags.values()) != 1:
                raise ValueError("Expected exactly one category")
            if label_path is None:
                if self.owner.label_widget.filename != path:
                    raise ValueError("Image no longer available")
                self.owner.checkbox_group.set_flags(flags)
                self.owner.save_current_flags()
            else:
                if os.path.exists(label_path):
                    with open(label_path, "rb") as stream:
                        content = stream.read()
                else:
                    content = None
                if content != self.files[label_path]:
                    raise ValueError("Annotation changed during processing")
                if content is None:
                    image = QImage(path)
                    if image.isNull():
                        raise ValueError("Cannot read image")
                    image_data = None
                    if self.owner.label_widget._config.get(
                        "store_data", False
                    ):
                        with open(path, "rb") as stream:
                            image_data = base64.b64encode(
                                stream.read()
                            ).decode("ascii")
                    data = create_xlabel_template(
                        image_path=os.path.relpath(
                            path, os.path.dirname(label_path)
                        ),
                        image_data=image_data,
                        image_height=image.height(),
                        image_width=image.width(),
                    )
                else:
                    data = json.loads(content)
                existing_flags = data.setdefault("flags", {})
                if any(
                    existing_flags.get(key) != value
                    for key, value in flags.items()
                ):
                    data["checked"] = False
                existing_flags.update(flags)
                save_json(data, label_path)
            self.succeeded += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.failed += 1
            self.owner._multitask_batch_errors.append(
                f"{os.path.basename(path)}: {exc}"
            )

    def update_progress(self):
        completed = self.total - len(self.queue) - len(self.running)
        self.progress.setLabelText(
            self.tr("Processed %d / %d images") % (completed, self.total)
        )
        self.progress.setValue(completed)
