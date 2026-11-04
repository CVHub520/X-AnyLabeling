"""Tests for X-AnyLabeling's headless inference CLI."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication

from anylabeling.services.auto_labeling.inference import (
    find_model,
    get_model_catalog,
    inspect_model,
    model_capabilities,
)
from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.views.common import cli_inference
from anylabeling.views.labeling.label_file import LabelFile
from anylabeling.views.labeling.shape import Shape


ROOT = Path(__file__).resolve().parents[2]


class FakeGroundingModel:
    """Stand-in with the same public methods used by the CLI model path."""

    def __init__(self, shapes: list[Shape]):
        self.config = {"type": "grounding_sam2"}
        self.shapes = shapes
        self.seen_prompt = None
        self.unloaded = False

    def predict_shapes(self, image, filename=None, text_prompt=None):
        self.seen_prompt = text_prompt
        return AutoLabelingResult(self.shapes)

    def set_auto_labeling_marks(self, marks):
        self.marks = marks

    def set_auto_labeling_conf(self, value):
        self.confidence = value

    def unload(self):
        self.unloaded = True


class TestModelCatalog(unittest.TestCase):
    """Verify registry discovery and no-download readiness inspection."""

    def test_list_contains_registered_grounding_sam2(self):
        with tempfile.TemporaryDirectory() as directory:
            models = get_model_catalog(Path(directory))
        model = next(
            item
            for item in models
            if item["model_id"] == "groundingdino_swint_sam2_large-r20240806"
        )
        capabilities = model_capabilities(model)
        self.assertTrue(capabilities["segmentation"])
        self.assertTrue(capabilities["text_prompt"])
        self.assertTrue(capabilities["text_prompt_required"])

    def test_unknown_model_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(Exception, "Unknown model ID"):
                find_model("missing-model", Path(directory))

    def test_check_reports_missing_weights_without_downloading(self):
        with tempfile.TemporaryDirectory() as directory:
            model = find_model(
                "groundingdino_swint_sam2_large-r20240806", Path(directory)
            )
            result = inspect_model(model, Path(directory), device="cpu")
        self.assertFalse(result["ready"])
        self.assertTrue(
            all(
                asset["requires_download"]
                for asset in result["assets"].values()
            )
        )


class TestCandidateExport(unittest.TestCase):
    """Exercise the CLI serialization contract using a fake XAL model."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.qt_app = QCoreApplication.instance() or QCoreApplication([])

    def _make_shape(self) -> Shape:
        shape = Shape(label="fox", shape_type="polygon")
        shape.add_point(QtCore.QPointF(2, 2))
        shape.add_point(QtCore.QPointF(14, 2))
        shape.add_point(QtCore.QPointF(14, 14))
        shape.add_point(QtCore.QPointF(2, 14))
        shape.score = 0.91
        return shape

    def _args(self, image_path: Path, output_path: Path, **overrides):
        values = {
            "input": str(image_path),
            "model": "groundingdino_swint_sam2_large-r20240806",
            "model_config": None,
            "config": None,
            "prompt": ["fox"],
            "marks_json": None,
            "device": "cpu",
            "confidence": None,
            "output": str(output_path),
            "recursive": False,
            "overwrite": False,
            "fail_fast": False,
            "json": True,
            "work_dir": str(output_path.parent / "xal-work"),
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _run_segment(self, args):
        fake_model = FakeGroundingModel([self._make_shape()])
        captured = io.StringIO()
        with patch.object(cli_inference, "load_model", return_value=fake_model):
            with contextlib.redirect_stdout(captured):
                exit_code = cli_inference.segment(args)
        return exit_code, json.loads(captured.getvalue()), fake_model

    def test_exports_reopenable_xlabel_and_separate_binary_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "fox.png"
            Image.new("RGB", (24, 18), "white").save(image_path)
            output_path = root / "review"

            exit_code, report, fake_model = self._run_segment(
                self._args(image_path, output_path)
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(fake_model.unloaded)
            self.assertEqual(fake_model.seen_prompt, "fox")
            item = report["results"][0]
            annotation_path = Path(item["outputs"]["annotation"])
            metadata_path = Path(item["outputs"]["metadata"])
            mask_path = Path(item["outputs"]["masks"][0])

            label_file = LabelFile(str(annotation_path))
            self.assertEqual(len(label_file.shapes), 1)
            self.assertEqual(label_file.shapes[0].label, "fox")

            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image)
            self.assertEqual(mask.shape, (18, 24))
            self.assertEqual(set(np.unique(mask)), {0, 255})
            self.assertEqual(int(mask[8, 8]), 255)
            self.assertEqual(int(mask[17, 23]), 0)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema"], "xanylabeling.candidates.v1")
            self.assertEqual(metadata["status"], "unreviewed_candidates")
            self.assertEqual(metadata["source"]["width"], 24)
            self.assertEqual(metadata["source"]["height"], 18)
            self.assertEqual(metadata["inference"]["prompts"], ["fox"])
            self.assertEqual(metadata["candidates"][0]["status"], "unreviewed")
            self.assertEqual(metadata["candidates"][0]["score"], 0.91)
            self.assertTrue(metadata["source"]["sha256"])

    def test_output_collision_fails_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "fox.png"
            Image.new("RGB", (24, 18), "white").save(image_path)
            output_path = root / "review"
            first = self._run_segment(self._args(image_path, output_path))
            self.assertEqual(first[0], 0)

            second = self._run_segment(self._args(image_path, output_path))
            self.assertEqual(second[0], 2)
            self.assertEqual(second[1]["status"], "failed")
            self.assertIn("--overwrite", second[1]["error"]["message"])
            replaced = self._run_segment(
                self._args(image_path, output_path, overwrite=True)
            )
            self.assertEqual(replaced[0], 0)

    def test_directory_results_follow_stable_relative_path_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "images"
            (input_dir / "z").mkdir(parents=True)
            Image.new("RGB", (24, 18), "white").save(input_dir / "z" / "b.png")
            Image.new("RGB", (24, 18), "white").save(input_dir / "a.png")
            exit_code, report, _ = self._run_segment(
                self._args(input_dir, root / "review", recursive=True)
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                [Path(item["input"]).relative_to(input_dir).as_posix() for item in report["results"]],
                ["a.png", "z/b.png"],
            )

    def test_marks_json_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            marks_path = Path(directory) / "marks.json"
            marks_path.write_text(
                json.dumps([{"type": "point", "data": [4, 8], "label": 1}]),
                encoding="utf-8",
            )
            self.assertEqual(
                cli_inference._read_marks(str(marks_path)),
                [{"type": "point", "data": [4, 8], "label": 1}],
            )
            marks_path.write_text("[]", encoding="utf-8")
            self.assertEqual(cli_inference._read_marks(str(marks_path)), [])


class TestCLIProcess(unittest.TestCase):
    """Exercise the actual argparse entry point as a subprocess."""

    def test_models_list_emits_one_json_document_without_gui(self):
        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "anylabeling.app",
                "models",
                "list",
                "--task",
                "segmentation",
                "--json",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        result = json.loads(completed.stdout)
        model_ids = {item["id"] for item in result["models"]}
        self.assertIn("groundingdino_swint_sam2_large-r20240806", model_ids)


if __name__ == "__main__":
    unittest.main()
