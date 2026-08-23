import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

PROFILE_VERSION = 1
METADATA_FILENAME = "metadata.json"
EMBEDDING_FILENAME = "embedding.npz"
MAX_EMBEDDING_FILE_SIZE = 64 * 1024 * 1024


class VisualPromptProfileError(ValueError):
    """Raised when a visual prompt profile is invalid or corrupted."""


class VisualPromptCompatibilityError(VisualPromptProfileError):
    """Raised when a profile cannot be used by the current model."""


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_name(name):
    if not isinstance(name, str) or not name.strip():
        raise VisualPromptProfileError("Profile name must not be empty.")
    name = name.strip()
    if len(name) > 128:
        raise VisualPromptProfileError(
            "Profile name must contain at most 128 characters."
        )
    if any(ord(character) < 32 for character in name):
        raise VisualPromptProfileError(
            "Profile name must not contain control characters."
        )
    if any(character in '<>:"/\\|?*' for character in name):
        raise VisualPromptProfileError(
            "Profile name contains characters that are invalid in a path."
        )
    return name


def _validate_json_value(value, path="metadata"):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise VisualPromptProfileError(
                f"{path} contains a non-finite number."
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise VisualPromptProfileError(
                    f"{path} contains a non-string key."
                )
            _validate_json_value(item, f"{path}.{key}")
        return
    raise VisualPromptProfileError(
        f"{path} contains an unsupported value: {type(value).__name__}."
    )


def _validate_created_at(created_at):
    try:
        parsed_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as error:
        raise VisualPromptProfileError(
            "Profile created_at must be a valid ISO 8601 timestamp."
        ) from error
    if parsed_time.tzinfo is None:
        raise VisualPromptProfileError(
            "Profile created_at must include a timezone."
        )


def _validate_model_signature(model_signature):
    required = {
        "model_type": str,
        "model_path": str,
        "model_file": str,
        "model_size": int,
        "architecture": str,
        "embedding_dimension": int,
    }
    if not isinstance(model_signature, dict):
        raise VisualPromptProfileError(
            "Profile model_signature must be an object."
        )
    for key, expected_type in required.items():
        value = model_signature.get(key)
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise VisualPromptProfileError(
                f"Profile model_signature.{key} is invalid."
            )
    _validate_json_value(model_signature, "model_signature")


def _validate_reference(reference):
    if not isinstance(reference, dict):
        raise VisualPromptProfileError(
            "Profile reference metadata must be an object."
        )
    if "image_path" not in reference or (
        reference["image_path"] is not None
        and (
            not isinstance(reference["image_path"], str)
            or not reference["image_path"].strip()
        )
    ):
        raise VisualPromptProfileError(
            "Profile reference image path is invalid."
        )
    boxes = reference.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        raise VisualPromptProfileError(
            "Profile reference boxes must not be empty."
        )
    boxes_array = np.asarray(boxes, dtype=np.float64)
    if (
        boxes_array.ndim != 2
        or boxes_array.shape[1] != 4
        or not np.isfinite(boxes_array).all()
        or np.any(boxes_array[:, 2:] <= boxes_array[:, :2])
    ):
        raise VisualPromptProfileError(
            "Profile reference boxes must be valid xyxy boxes."
        )
    image_size = reference.get("image_size")
    valid_size = (
        isinstance(image_size, list)
        and len(image_size) == 2
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and np.isfinite(value)
            and value > 0
            for value in image_size
        )
    )
    boxes_in_bounds = valid_size and (
        np.all(boxes_array[:, :2] >= 0)
        and np.all(boxes_array[:, 2] <= image_size[0])
        and np.all(boxes_array[:, 3] <= image_size[1])
    )
    if not boxes_in_bounds:
        raise VisualPromptProfileError(
            "Profile reference boxes exceed the reference image."
        )
    instance_count = reference.get("instance_count")
    if instance_count is not None and instance_count != len(boxes):
        raise VisualPromptProfileError(
            "Profile reference instance count does not match its boxes."
        )
    _validate_json_value(reference, "reference")


def _validate_classes(classes):
    if (
        not isinstance(classes, list)
        or not classes
        or any(
            not isinstance(item, str) or not item.strip() for item in classes
        )
    ):
        raise VisualPromptProfileError(
            "Profile classes must be non-empty strings."
        )


def _validate_vpe(vpe, classes, embedding_dimension):
    if not isinstance(vpe, np.ndarray):
        raise VisualPromptProfileError(
            "Profile embedding must be a NumPy array."
        )
    if vpe.dtype != np.float32:
        raise VisualPromptProfileError(
            "Profile embedding dtype must be float32."
        )
    if (
        vpe.ndim != 3
        or vpe.shape[0] != 1
        or vpe.shape[1] != len(classes)
        or vpe.shape[2] != embedding_dimension
    ):
        raise VisualPromptProfileError(
            "Profile embedding shape is inconsistent with its metadata."
        )
    if not np.isfinite(vpe).all():
        raise VisualPromptProfileError(
            "Profile embedding contains non-finite values."
        )


@dataclass(frozen=True)
class VisualPromptProfile:
    """Portable YOLOE visual prompt metadata and embedding."""

    name: str
    created_at: str
    model_name: str
    model_signature: dict
    reference: dict
    classes: list
    vpe: np.ndarray
    version: int = PROFILE_VERSION

    def validate(self):
        name = _validate_name(self.name)
        if isinstance(self.version, bool) or self.version != PROFILE_VERSION:
            raise VisualPromptProfileError(
                "Unsupported visual prompt profile version: "
                f"{self.version}; expected {PROFILE_VERSION}."
            )
        _validate_created_at(self.created_at)
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise VisualPromptProfileError("Profile model_name is missing.")
        _validate_model_signature(self.model_signature)
        _validate_reference(self.reference)
        _validate_classes(self.classes)
        reference_classes = self.reference.get("classes")
        if reference_classes is not None and reference_classes != self.classes:
            raise VisualPromptProfileError(
                "Profile classes do not match its reference metadata."
            )
        _validate_vpe(
            self.vpe,
            self.classes,
            self.model_signature["embedding_dimension"],
        )
        return name

    def to_metadata(self, embedding_sha256):
        name = self.validate()
        return {
            "version": self.version,
            "name": name,
            "created_at": self.created_at,
            "model_name": self.model_name,
            "model_signature": self.model_signature,
            "reference": self.reference,
            "classes": self.classes,
            "vpe_shape": list(self.vpe.shape),
            "vpe_dtype": str(self.vpe.dtype),
            "embedding_file": EMBEDDING_FILENAME,
            "embedding_sha256": embedding_sha256,
        }

    def save(self, directory):
        """Atomically write embedding first and metadata as commit marker."""
        self.validate()
        directory = os.path.abspath(os.fspath(directory))
        os.makedirs(directory, exist_ok=True)
        embedding_path = os.path.join(directory, EMBEDDING_FILENAME)
        metadata_path = os.path.join(directory, METADATA_FILENAME)
        embedding_temp = None
        metadata_temp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".npz", dir=directory, delete=False
            ) as file:
                embedding_temp = file.name
                np.savez_compressed(file, vpe=self.vpe)
            embedding_sha256 = _sha256(embedding_temp)
            metadata = self.to_metadata(embedding_sha256)
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                dir=directory,
                encoding="utf-8",
                delete=False,
            ) as file:
                metadata_temp = file.name
                json.dump(metadata, file, ensure_ascii=False, indent=2)
                file.write("\n")
            os.replace(embedding_temp, embedding_path)
            embedding_temp = None
            os.replace(metadata_temp, metadata_path)
            metadata_temp = None
        finally:
            for path in (embedding_temp, metadata_temp):
                if path and os.path.exists(path):
                    os.unlink(path)
        return metadata_path

    @classmethod
    def create(
        cls, name, model_name, model_signature, reference, classes, vpe
    ):
        created_at = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        return cls(
            name=name,
            created_at=created_at,
            model_name=model_name,
            model_signature=dict(model_signature),
            reference=dict(reference),
            classes=list(classes),
            vpe=np.asarray(vpe, dtype=np.float32),
        )

    @classmethod
    def load(cls, path):
        path = os.path.abspath(os.fspath(path))
        metadata_path = (
            os.path.join(path, METADATA_FILENAME)
            if os.path.isdir(path)
            else path
        )
        if not os.path.isfile(metadata_path):
            raise VisualPromptProfileError(
                f"Visual prompt metadata does not exist: {metadata_path}"
            )
        try:
            with open(metadata_path, encoding="utf-8") as file:
                metadata = json.load(file)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise VisualPromptProfileError(
                f"Could not read visual prompt metadata: {error}"
            ) from error
        if not isinstance(metadata, dict):
            raise VisualPromptProfileError(
                "Visual prompt metadata must contain a JSON object."
            )
        required_keys = {
            "version",
            "name",
            "created_at",
            "model_name",
            "model_signature",
            "reference",
            "classes",
            "vpe_shape",
            "vpe_dtype",
            "embedding_file",
            "embedding_sha256",
        }
        missing = sorted(required_keys - metadata.keys())
        if missing:
            raise VisualPromptProfileError(
                "Visual prompt metadata is missing: " + ", ".join(missing)
            )
        if metadata["version"] != PROFILE_VERSION:
            raise VisualPromptProfileError(
                "Unsupported visual prompt profile version: "
                f"{metadata['version']}; expected {PROFILE_VERSION}."
            )
        if metadata["embedding_file"] != EMBEDDING_FILENAME:
            raise VisualPromptProfileError(
                "Visual prompt embedding filename is invalid."
            )
        embedding_path = os.path.join(
            os.path.dirname(metadata_path), EMBEDDING_FILENAME
        )
        if not os.path.isfile(embedding_path):
            raise VisualPromptProfileError(
                f"Visual prompt embedding does not exist: {embedding_path}"
            )
        if os.path.getsize(embedding_path) > MAX_EMBEDDING_FILE_SIZE:
            raise VisualPromptProfileError(
                "Visual prompt embedding file is unexpectedly large."
            )
        checksum = metadata["embedding_sha256"]
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or _sha256(embedding_path) != checksum.lower()
        ):
            raise VisualPromptProfileError(
                "Visual prompt embedding checksum is invalid."
            )
        try:
            with np.load(embedding_path, allow_pickle=False) as archive:
                if archive.files != ["vpe"]:
                    raise VisualPromptProfileError(
                        "Visual prompt embedding archive is invalid."
                    )
                vpe = archive["vpe"].copy()
        except VisualPromptProfileError:
            raise
        except (OSError, ValueError, KeyError) as error:
            raise VisualPromptProfileError(
                f"Could not read visual prompt embedding: {error}"
            ) from error
        if metadata["vpe_shape"] != list(vpe.shape):
            raise VisualPromptProfileError(
                "Visual prompt embedding shape does not match metadata."
            )
        if metadata["vpe_dtype"] != str(vpe.dtype):
            raise VisualPromptProfileError(
                "Visual prompt embedding dtype does not match metadata."
            )
        profile = cls(
            name=metadata["name"],
            created_at=metadata["created_at"],
            model_name=metadata["model_name"],
            model_signature=metadata["model_signature"],
            reference=metadata["reference"],
            classes=metadata["classes"],
            vpe=vpe,
            version=metadata["version"],
        )
        profile.validate()
        return profile

    def validate_compatibility(self, current_signature):
        """Require the current YOLOE architecture and weights to match."""
        keys = (
            "model_type",
            "model_file",
            "model_size",
            "architecture",
            "embedding_dimension",
        )
        mismatches = [
            key
            for key in keys
            if self.model_signature.get(key) != current_signature.get(key)
        ]
        if mismatches:
            details = ", ".join(
                f"{key}: {self.model_signature.get(key)!r} != "
                f"{current_signature.get(key)!r}"
                for key in mismatches
            )
            raise VisualPromptCompatibilityError(
                "This visual prompt is incompatible with the current model "
                f"({details}). Regenerate it from the saved reference."
            )
        return True
