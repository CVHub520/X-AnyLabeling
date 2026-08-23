import hashlib
import json
import os
import tempfile
import unittest

import numpy as np

from anylabeling.services.auto_labeling.visual_prompt_profile import (
    EMBEDDING_FILENAME,
    METADATA_FILENAME,
    VisualPromptCompatibilityError,
    VisualPromptProfile,
    VisualPromptProfileError,
)


class TestVisualPromptProfile(unittest.TestCase):
    @staticmethod
    def make_profile(name="product_A"):
        return VisualPromptProfile.create(
            name=name,
            model_name="yoloe-11s-seg",
            model_signature={
                "model_type": "yoloe",
                "model_name": "yoloe-11s-seg",
                "model_path": "/models/yoloe-11s-seg.pt",
                "model_file": "yoloe-11s-seg.pt",
                "model_size": 123456,
                "architecture": "YOLOEModel",
                "embedding_dimension": 512,
            },
            reference={
                "image_path": "/data/reference.jpg",
                "image_size": [640, 480],
                "boxes": [[10.0, 20.0, 100.0, 120.0]],
                "instance_count": 1,
                "classes": ["object"],
            },
            classes=["object"],
            vpe=np.ones((1, 1, 512), dtype=np.float32),
        )

    def test_round_trip_uses_json_and_npz_without_pickle(self):
        profile = self.make_profile()
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = profile.save(directory)
            restored = VisualPromptProfile.load(metadata_path)

            self.assertEqual(
                metadata_path, os.path.join(directory, METADATA_FILENAME)
            )
            self.assertEqual(restored.name, "product_A")
            self.assertEqual(restored.reference, profile.reference)
            self.assertEqual(restored.vpe.dtype, np.float32)
            np.testing.assert_array_equal(restored.vpe, profile.vpe)
            with np.load(
                os.path.join(directory, EMBEDDING_FILENAME),
                allow_pickle=False,
            ) as archive:
                self.assertEqual(archive.files, ["vpe"])

    def test_rejects_invalid_profile_metadata(self):
        invalid_profiles = (
            self.make_profile("../escape"),
            VisualPromptProfile(
                **{**self.make_profile().__dict__, "version": 99}
            ),
            VisualPromptProfile(
                **{
                    **self.make_profile().__dict__,
                    "vpe": np.ones((1, 1, 512), dtype=np.float64),
                }
            ),
            VisualPromptProfile(
                **{
                    **self.make_profile().__dict__,
                    "reference": {"boxes": [[1, 1, 0, 2]]},
                }
            ),
        )
        for profile in invalid_profiles:
            with self.subTest(name=profile.name, version=profile.version):
                with self.assertRaises(VisualPromptProfileError):
                    profile.validate()

    def test_rejects_unsupported_version_before_loading_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_profile().save(directory)
            metadata_path = os.path.join(directory, METADATA_FILENAME)
            with open(metadata_path, encoding="utf-8") as file:
                metadata = json.load(file)
            metadata["version"] = 2
            with open(metadata_path, "w", encoding="utf-8") as file:
                json.dump(metadata, file)

            with self.assertRaisesRegex(
                VisualPromptProfileError, "Unsupported.*version"
            ):
                VisualPromptProfile.load(directory)

    def test_rejects_corrupted_embedding_and_preserves_no_object_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_profile().save(directory)
            embedding_path = os.path.join(directory, EMBEDDING_FILENAME)
            with open(embedding_path, "ab") as file:
                file.write(b"corruption")

            with self.assertRaisesRegex(VisualPromptProfileError, "checksum"):
                VisualPromptProfile.load(directory)

    def test_rejects_archive_shape_or_dtype_mismatch(self):
        for vpe in (
            np.ones((1, 1, 256), dtype=np.float32),
            np.ones((1, 1, 512), dtype=np.float64),
        ):
            with self.subTest(shape=vpe.shape, dtype=vpe.dtype):
                with tempfile.TemporaryDirectory() as directory:
                    self.make_profile().save(directory)
                    embedding_path = os.path.join(
                        directory, EMBEDDING_FILENAME
                    )
                    np.savez_compressed(embedding_path, vpe=vpe)
                    with open(embedding_path, "rb") as file:
                        checksum = hashlib.sha256(file.read()).hexdigest()
                    metadata_path = os.path.join(directory, METADATA_FILENAME)
                    with open(metadata_path, encoding="utf-8") as file:
                        metadata = json.load(file)
                    metadata["embedding_sha256"] = checksum
                    with open(metadata_path, "w", encoding="utf-8") as file:
                        json.dump(metadata, file)

                    with self.assertRaises(VisualPromptProfileError):
                        VisualPromptProfile.load(directory)

    def test_model_compatibility_checks_weights_architecture_and_dimension(
        self,
    ):
        profile = self.make_profile()
        compatible = dict(profile.model_signature)
        self.assertTrue(profile.validate_compatibility(compatible))

        for key, value in (
            ("model_type", "other"),
            ("model_file", "yoloe-11l-seg.pt"),
            ("model_size", 999),
            ("architecture", "OtherModel"),
            ("embedding_dimension", 256),
        ):
            with self.subTest(key=key):
                incompatible = dict(compatible)
                incompatible[key] = value
                with self.assertRaises(VisualPromptCompatibilityError):
                    profile.validate_compatibility(incompatible)


if __name__ == "__main__":
    unittest.main()
