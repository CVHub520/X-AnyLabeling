"""Preflight and per-file output protection for ellipse conversion paths."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import functools
import inspect
from typing import Any, Iterator, Mapping, Sequence, TextIO
from xml.etree import ElementTree

from .ellipse import (
    ellipse_direction,
    segmentation_points,
    validate_polygon_points,
)


class ExportPreflightError(ValueError):
    """Whole-batch validation failure with individual source diagnostics."""

    def __init__(self, failures: dict[str, str]) -> None:
        """Keep structured failures for GUI and CLI outcome reports.

        Args:
            failures: Source paths mapped to their validation errors.
        """
        self.failures = failures
        super().__init__(
            "Export preflight failed; no outputs changed:\n"
            + "\n".join(f"{path}: {error}" for path, error in failures.items())
        )


def report_export(source_parameter: str | None = None) -> Any:
    """Record per-source outcomes without changing converter return values.

    Args:
        source_parameter: Input filename parameter, or None for a summary file.

    Returns:
        A decorator preserving the existing export call signature.
    """

    def decorate(function: Any) -> Any:
        """Wrap one converter operation with outcome recording.

        Args:
            function: Existing native-to-external conversion method.

        Returns:
            The wrapped method.
        """
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Run an export and retain its success, skip or failure reason.

            Args:
                self: Label converter owning the batch report.
                args: Original positional arguments.
                kwargs: Original keyword arguments.

            Returns:
                The original converter result.
            """
            bound = signature.bind(self, *args, **kwargs)
            source = (
                str(bound.arguments[source_parameter])
                if source_parameter
                else None
            )
            if source is not None:
                self.export_results.setdefault(source, ("unprocessed", ""))
            try:
                result = function(self, *args, **kwargs)
            except InterruptedError:
                raise
            except Exception as error:
                targets = (
                    [source]
                    if source
                    else [
                        p
                        for p, (status, _) in self.export_results.items()
                        if status == "unprocessed"
                    ]
                )
                for path in targets:
                    self.export_results[path] = ("failed", str(error))
                raise
            targets = (
                [source]
                if source
                else [
                    p
                    for p, (status, _) in self.export_results.items()
                    if status == "unprocessed"
                ]
            )
            skipped = result is True and bound.arguments.get(
                "skip_empty_files", False
            )
            for path in targets:
                self.export_results[path] = (
                    "skipped" if skipped else "success",
                    "empty annotation" if skipped else "",
                )
            return result

        return wrapped

    return decorate


def export_report(results: Mapping[str, tuple[str, str]]) -> str:
    """Format a shared GUI/CLI batch report.

    Args:
        results: Per-source statuses and diagnostic reasons.

    Returns:
        Counts followed by every selected source and its outcome.
    """
    statuses = ("success", "failed", "skipped", "unprocessed")
    counts = ", ".join(
        f"{status}={sum(value[0] == status for value in results.values())}"
        for status in statuses
    )
    return (
        counts
        + "\n"
        + "\n".join(
            f"{path}: {status}" + (f" — {reason}" if reason else "")
            for path, (status, reason) in results.items()
        )
    )


def preflight_shapes(
    data: Mapping[str, Any],
    source: str,
    mode: str,
    count: int = 128,
    classes: Sequence[str] | None = None,
    mapping: Mapping[str, Any] | None = None,
    ignored_labels: Sequence[str] = (),
) -> None:
    """Validate newly supported objects without changing legacy shape rules.

    Args:
        data: Native annotation document.
        source: Input path for actionable diagnostics.
        mode: 'seg' for supported segmentation, 'native' for conversion,
            or a descriptive name for an unsupported export mode.
        count: Current ellipse sampling count.
        classes: Optional export class list.
        mapping: Optional mask mapping table.
        ignored_labels: Existing format-specific ignore labels.

    Raises:
        ValueError: If a new object cannot be represented by this operation.
    """
    validate_polygon_points(count)
    for index, shape in enumerate(data.get("shapes", [])):
        if shape.get("label") in ignored_labels:
            continue
        kind = shape.get("shape_type")
        if kind != "ellipse" and not (
            kind == "quadrilateral" and mode == "seg"
        ):
            continue
        context = f"{source}: object {index} ({shape.get('label', '')})"
        try:
            if kind == "ellipse":
                ellipse_direction(shape, context)
                if mode not in ("seg", "native"):
                    raise ValueError(
                        f"ellipse is incompatible with {mode}; convert explicitly to polygon first"
                    )
            if mode == "seg":
                width, height = data["imageWidth"], data["imageHeight"]
                if (
                    type(width) is not int
                    or type(height) is not int
                    or width <= 0
                    or height <= 0
                ):
                    raise ValueError(
                        "image dimensions must be positive integers"
                    )
                points = segmentation_points(shape, width, height, count)
                if kind == "quadrilateral":
                    area = (
                        abs(
                            sum(
                                a[0] * b[1] - b[0] * a[1]
                                for a, b in zip(
                                    points, points[1:] + points[:1]
                                )
                            )
                        )
                        / 2
                    )
                    if area <= 1e-6:
                        raise ValueError(
                            "quadrilateral degenerates after legacy boundary clamping"
                        )
                if classes is not None and shape.get("label") not in classes:
                    raise ValueError("label is missing from export classes")
                if mapping is not None and shape.get(
                    "label"
                ) not in mapping.get("colors", {}):
                    raise ValueError(
                        "label is missing from mask color mapping"
                    )
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise ValueError(f"{context}: {error}") from error


def preflight_files(
    paths: Sequence[str],
    mode: str,
    count: int = 128,
    classes: Sequence[str] | None = None,
    mapping: Mapping[str, Any] | None = None,
    ignored_labels: Sequence[str] = (),
) -> dict[str, str]:
    """Preflight every selected input before touching existing outputs.

    Args:
        paths: Native annotation paths; missing annotations keep legacy handling.
        mode: Operation mode passed to preflight_shapes.
        count: Current ellipse sampling count.
        classes: Optional class mapping.
        mapping: Optional mask color mapping.
        ignored_labels: Existing format-specific ignore labels.

    Returns:
        SHA-256 digests for revalidation before each output is committed.

    Raises:
        ValueError: If any selected input fails preflight.
    """
    fingerprints = {}
    errors = {}
    for path in dict.fromkeys(map(str, paths)):
        if not os.path.isfile(path):
            continue
        try:
            raw = Path(path).read_bytes()
            preflight_shapes(
                json.loads(raw),
                path,
                mode,
                count,
                classes,
                mapping,
                ignored_labels,
            )
            fingerprints[path] = hashlib.sha256(raw).hexdigest()
        except (ValueError, TypeError, KeyError, OSError) as error:
            errors[path] = str(error)
    if errors:
        raise ExportPreflightError(errors)
    return fingerprints


def verify_inputs(fingerprints: Mapping[str, str]) -> None:
    """Reject input changes after preflight.

    Args:
        fingerprints: Expected source hashes for the pending output.

    Raises:
        ValueError: If a source changed or disappeared during conversion.
    """
    for path, expected in fingerprints.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"{path}: input changed after preflight; run export again"
            )


@contextmanager
def atomic_open(
    path: str,
    mode: str = "w",
    encoding: str = "utf-8",
    fingerprints: Mapping[str, str] | None = None,
    skip_empty: bool = False,
    allow_source_replace: bool = False,
) -> Iterator[TextIO]:
    """Prepare a complete text file and replace its destination atomically.

    Args:
        path: Destination file; its existing parent must be writable.
        mode: Text write mode, retained for familiar open-style call sites.
        encoding: Output encoding.
        fingerprints: Optional inputs to check immediately before replacement.
        skip_empty: Leave an existing destination untouched for empty text.
        allow_source_replace: Only explicit native conversion may replace input.

    Yields:
        An open stream backed by a same-directory temporary file.

    Raises:
        ValueError: If generated JSON/XML or input fingerprints are invalid.
        OSError: If writing or replacing fails; the old destination survives.
    """
    destination = os.path.abspath(path)
    if not allow_source_replace and any(
        os.path.normcase(os.path.realpath(source))
        == os.path.normcase(os.path.realpath(destination))
        for source in (fingerprints or {})
    ):
        raise ValueError(
            f"{path}: export destination overlaps a source annotation"
        )
    descriptor, temporary = tempfile.mkstemp(
        prefix=".xal_ellipse_", suffix=".tmp", dir=os.path.dirname(destination)
    )
    try:
        with os.fdopen(descriptor, mode, encoding=encoding) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        if destination.lower().endswith(".json"):
            with open(temporary, encoding=encoding) as stream:
                json.load(stream)
        elif destination.lower().endswith(".xml"):
            ElementTree.parse(temporary)
        verify_inputs(fingerprints or {})
        if not (skip_empty and os.path.getsize(temporary) == 0):
            os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def atomic_bytes(
    path: str, content: bytes, fingerprints: Mapping[str, str] | None = None
) -> None:
    """Safely replace a completely encoded binary output.

    Args:
        path: Destination path.
        content: Complete encoded file contents.
        fingerprints: Inputs to check before replacement.

    Raises:
        OSError: If writing or replacement fails.
        ValueError: If input data changed after preflight.
    """
    descriptor, temporary = tempfile.mkstemp(
        prefix=".xal_ellipse_",
        suffix=".tmp",
        dir=os.path.dirname(os.path.abspath(path)),
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        verify_inputs(fingerprints or {})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
