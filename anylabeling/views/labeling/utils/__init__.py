def __getattr__(name: str):
    if name in {
        "gradient_text",
        "hex_to_rgb",
        "is_chinese",
        "find_most_similar_label",
    }:
        from . import general as _general

        return getattr(_general, name)

    if name in {
        "Struct",
        "add_actions",
        "scan_all_images",
        "distance",
        "distance_to_line",
        "fmt_shortcut",
        "label_validator",
        "new_action",
        "new_button",
        "new_icon",
        "new_icon_path",
        "on_thumbnail_click",
    }:
        from . import qt as _qt

        return getattr(_qt, name)

    if name in {"AsyncExifScanner", "ExifProcessingDialog"}:
        from . import async_exif as _async_exif

        return getattr(_async_exif, name)

    if name in {
        "check_img_exif",
        "get_pil_img_dim",
        "img_arr_to_b64",
        "img_b64_to_arr",
        "img_data_to_arr",
        "img_data_to_pil",
        "img_data_to_png_data",
        "img_pil_to_data",
        "process_image_exif",
    }:
        from . import image as _image

        return getattr(_image, name)

    if name == "io_open":
        from ._io import io_open

        return io_open

    if name == "label_colormap":
        from .colormap import label_colormap

        return label_colormap

    if name == "open_video_file":
        from .video import open_video_file

        return open_video_file

    if name == "run_all_images":
        from .batch import run_all_images

        return run_all_images

    if name == "save_crop":
        from .crop import save_crop

        return save_crop

    if name in {
        "masks_to_bboxes",
        "polygons_to_mask",
        "shape_to_mask",
        "shapes_to_label",
        "rectangle_from_diagonal",
        "shape_conversion",
    }:
        from . import shape as _shape

        return getattr(_shape, name)

    if name in {
        "export_yolo_annotation",
        "export_voc_annotation",
        "export_coco_annotation",
        "export_dota_annotation",
        "export_mask_annotation",
        "export_mot_annotation",
        "export_odvg_annotation",
        "export_pporc_annotation",
        "export_vlm_r1_ovd_annotation",
    }:
        from . import export as _export

        return getattr(_export, name)

    if name in {
        "upload_image_flags_file",
        "upload_label_flags_file",
        "upload_shape_attrs_file",
        "upload_label_classes_file",
        "upload_yolo_annotation",
        "upload_voc_annotation",
        "upload_coco_annotation",
        "upload_dota_annotation",
        "upload_mask_annotation",
        "upload_mot_annotation",
        "upload_odvg_annotation",
        "upload_mmgd_annotation",
        "upload_ppocr_annotation",
        "upload_vlm_r1_ovd_annotation",
    }:
        from . import upload as _upload

        return getattr(_upload, name)

    raise AttributeError(name)
