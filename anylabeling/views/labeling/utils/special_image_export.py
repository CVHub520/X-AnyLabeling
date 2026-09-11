"""Non-destructive export of image/annotation pairs selected for review."""

import json
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

EXPORT_MARK_FIELD = "export_marked"


class ExportCancelled(Exception):
    """Signal cancellation without treating it as an export failure."""


def read_export_mark(label_file: str) -> bool:
    """Read a persisted mark, treating missing or invalid labels as unmarked.

    Args:
        label_file: Annotation JSON path.

    Returns:
        Whether the top-level export mark is exactly true.
    """
    try:
        with open(label_file, encoding="utf-8") as stream:
            data = json.load(stream)
        return isinstance(data, dict) and data.get(EXPORT_MARK_FIELD) is True
    except (OSError, ValueError):
        return False


def export_marked_pairs(
    image_files: list[str],
    source_dir: str,
    destination: str,
    output_dir: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Copy marked pairs without overwriting files or changing source labels.

    Validate every destination before copying. Preserve relative directories,
    and rewrite only the exported JSON's imagePath to its adjacent image.
    On a copy error or cancellation, remove files created by this invocation.
    Empty directories may remain. Missing annotations are not marked; invalid
    annotations fail explicitly rather than silently omitting possible marks.

    Args:
        image_files: Unfiltered images in the source folder.
        source_dir: Root used to preserve relative image paths.
        destination: Export directory outside the source tree.
        output_dir: Optional flat annotation directory used by the application.
        progress: Callback after each scanned/copied pair; may raise
            ExportCancelled. Total covers scanning and copying.

    Returns:
        Number of copied image/annotation pairs.

    Raises:
        ValueError: An unsafe path, ambiguous label, or invalid JSON is found.
        OSError: A destination exists or a read/write operation fails.
        ExportCancelled: The progress callback requests cancellation.
    """
    root = Path(source_dir).resolve()
    target_root = Path(destination).resolve()
    if target_root == root or root in target_root.parents:
        raise ValueError(
            "Choose an export directory outside the source folder."
        )
    plan = []
    targets = set()
    labels = set()
    total = len(image_files) * 2
    for index, image_file in enumerate(image_files):
        if progress:
            progress(index, total)
        image = Path(image_file).resolve()
        try:
            relative = image.relative_to(root)
        except ValueError:
            # Skip images that live outside the source root; the caller only
            # promises "unfiltered images in the source folder" but external
            # symlinks or stale entries can still slip through scan_all_images.
            continue
        label = image.with_suffix(".json")
        if output_dir:
            label = Path(output_dir).resolve() / label.name
        if not label.exists():
            continue
        with label.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid annotation: {label}")
        if data.get(EXPORT_MARK_FIELD) is not True:
            continue
        if label in labels:
            raise ValueError(
                f"Multiple images share the same annotation: {label}"
            )
        labels.add(label)
        image_target = (target_root / relative).resolve()
        label_target = image_target.with_suffix(".json")
        for target in (image_target, label_target):
            if target_root not in target.parents:
                raise ValueError(
                    f"Destination escapes the export directory: {target}"
                )
            if target in targets or os.path.lexists(target):
                raise FileExistsError(
                    f"Export would overwrite a file: {target}"
                )
            targets.add(target)
        data["imagePath"] = image_target.name
        plan.append((image, image_target, label_target, data))

    created = []
    try:
        for index, (image, image_target, label_target, data) in enumerate(
            plan
        ):
            if progress:
                progress(len(image_files) + index, total)
            image_target.parent.mkdir(parents=True, exist_ok=True)
            with image.open("rb") as source, image_target.open("xb") as target:
                created.append(image_target)
                shutil.copyfileobj(source, target)
            with label_target.open("x", encoding="utf-8") as stream:
                created.append(label_target)
                json.dump(data, stream, ensure_ascii=False, indent=2)
        if progress:
            progress(total, total)
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return len(plan)
