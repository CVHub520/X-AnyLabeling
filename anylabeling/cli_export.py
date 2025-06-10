import os.path as osp
from functools import partial

from tqdm import tqdm

from anylabeling.views.labeling.label_converter import LabelConverter
from anylabeling.views.labeling.logger import logger

SUPPORTED_EXPORT_FORMATS = {
    "yolo": ["hbb", "obb", "seg", "pose"]
}


def export_annotations(args, image_paths, json_input_dir, export_output_dir):
    export_format = args.export_format.lower()

    converter = LabelConverter()
    converter.classes = args.labels

    try:
        format_type, mode = export_format.split(":")
    except ValueError:
        raise ValueError(f"Invalid export_format: '{export_format}', expected format like 'yolo:hbb'")

    # Check supported formats
    if format_type not in SUPPORTED_EXPORT_FORMATS:
        supported = ", ".join(f"{f}:{m}" for f, modes in SUPPORTED_EXPORT_FORMATS.items() for m in modes)
        raise ValueError(f"Unsupported export format: '{format_type}'. Supported formats: {supported}")

    if mode not in SUPPORTED_EXPORT_FORMATS[format_type]:
        supported_modes = ", ".join(SUPPORTED_EXPORT_FORMATS[format_type])
        raise ValueError(f"Unsupported mode '{mode}' for format '{format_type}'. "
                         f"Supported modes: {supported_modes}")

    # Partial function to freeze export settings
    if format_type == "yolo":
        convert_annotation = partial(converter.custom_to_yolo, mode=mode, skip_empty_files=True)
    else:
        # Placeholder for future support
        raise NotImplementedError(f"Export format '{format_type}' is not yet implemented.")

    logger.info(f"Exporting annotations in format '{format_type}:{mode}' to: {export_output_dir}")

    for image_path in tqdm(image_paths, desc="Exporting", unit="file"):
        image_name = osp.basename(image_path)
        base_name = osp.splitext(image_name)[0]
        json_label_path = osp.join(json_input_dir, f"{base_name}.json")
        export_label_path = osp.join(export_output_dir, f"{base_name}.txt")

        convert_annotation(json_label_path, export_label_path)

    logger.info("Export completed successfully.")
