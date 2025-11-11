_CUSTOM_MODELS = [
    "remote_server",
    "florence2",
    "doclayout_yolo",
    "open_vision",
    "segment_anything",
    "segment_anything_2",
    "segment_anything_2_video",
    "sam_med2d",
    "sam_hq",
    "yolov5",
    "yolov6",
    "yolov7",
    "yolov8",
    "yolov8_seg",
    "yolox",
    "yolov5_resnet",
    "yolov6_face",
    "rtdetr",
    "yolo_nas",
    "yolox_dwpose",
    "clrnet",
    "ppocr_v4",
    "yolov5_sam",
    "yolov8_sam2",
    "efficientvit_sam",
    "yolov5_track",
    "damo_yolo",
    "yolov5_sahi",
    "yolov8_sahi",
    "grounding_sam",
    "grounding_sam2",
    "grounding_dino",
    "grounding_dino_api",
    "yolov5_obb",
    "gold_yolo",
    "ram",
    "yolov5_seg",
    "yolov5_ram",
    "yolov8_pose",
    "pulc_attribute",
    "internimage_cls",
    "edge_sam",
    "yolov5_cls",
    "yolov8_cls",
    "yolov8_obb",
    "yolov5_car_plate",
    "rtmdet_pose",
    "yolov9",
    "yolow",
    "yolov10",
    "rmbg",
    "depth_anything",
    "depth_anything_v2",
    "yolow_ram",
    "rtdetrv2",
    "yolov8_det_track",
    "yolov8_seg_track",
    "yolov8_obb_track",
    "yolov8_pose_track",
    "yolo11",
    "yolo11_cls",
    "yolo11_obb",
    "yolo11_seg",
    "yolo11_pose",
    "yolo11_det_track",
    "yolo11_seg_track",
    "yolo11_obb_track",
    "yolo11_pose_track",
    "upn",
    "geco",
    "rfdetr",
    "rfdetr_seg",
    "dfine",
    "yolo12",
    "u_rtdetr",
    "yoloe",
    "ppocr_v5",
    "deimv2",
]


# --- set_cache_auto_label ---
_CACHED_AUTO_LABELING_MODELS = [
    "segment_anything_2_video",
]


# --- set_auto_labeling_marks ---
_AUTO_LABELING_MARKS_MODELS = [
    "remote_server",
    "segment_anything",
    "segment_anything_2",
    "segment_anything_2_video",
    "sam_med2d",
    "sam_hq",
    "yolov5_sam",
    "efficientvit_sam",
    "grounding_sam",
    "grounding_sam2",
    "open_vision",
    "edge_sam",
    "florence2",
    "geco",
    "yoloe",
]


# --- set_mask_fineness ---
_AUTO_LABELING_MASK_FINENESS_MODELS = [
    "remote_server",
    "segment_anything",
    "segment_anything_2",
    "segment_anything_2_video",
    "sam_med2d",
    "sam_hq",
    "yolov5_sam",
    "efficientvit_sam",
    "grounding_sam",
    "grounding_sam2",
    "edge_sam",
    "rfdetr_seg",
]


# --- skip detection step ---
_SKIP_DET_MODELS = [
    "ppocr_v4",
    "ppocr_v5",
]


# --- skip_prediction_on_new_marks ---
_SKIP_PREDICTION_ON_NEW_MARKS_MODELS = [
    "yoloe",
]


# --- set_auto_labeling_api_token ---
_AUTO_LABELING_API_TOKEN_MODELS = [
    "remote_server",
    "grounding_dino_api",
]


# --- set_auto_labeling_reset_tracker ---
_AUTO_LABELING_RESET_TRACKER_MODELS = [
    "remote_server",
    "yolov5_det_track",
    "yolov8_det_track",
    "yolov8_obb_track",
    "yolov8_seg_track",
    "yolov8_pose_track",
    "segment_anything_2_video",
    "yolo11_det_track",
    "yolo11_seg_track",
    "yolo11_obb_track",
    "yolo11_pose_track",
]


# --- set_auto_labeling_conf ---
_AUTO_LABELING_CONF_MODELS = [
    "remote_server",
    "upn",
    "damo_yolo",
    "gold_yolo",
    "grounding_dino",
    "grounding_dino_api",
    "rtdetr",
    "rtdetrv2",
    "yolo_nas",
    "yolov5_obb",
    "yolov5_seg",
    "yolov5_det_track",
    "yolov5",
    "yolov6",
    "yolov6_face",
    "yolov7",
    "yolov8_sam2",
    "yolov8_obb",
    "yolov8_pose",
    "yolov8_seg",
    "yolov8_det_track",
    "yolov8_seg_track",
    "yolov8_obb_track",
    "yolov8_pose_track",
    "yolov8",
    "yolov9",
    "yolov10",
    "yolo11",
    "yolo11_obb",
    "yolo11_seg",
    "yolo11_pose",
    "yolo11_det_track",
    "yolo11_seg_track",
    "yolo11_obb_track",
    "yolo11_pose_track",
    "yolow",
    "yolox",
    "doclayout_yolo",
    "rfdetr",
    "rfdetr_seg",
    "deimv2",
    "dfine",
    "yolo12",
    "u_rtdetr",
    "yoloe",
    "grounding_sam2",
]


# --- set_auto_labeling_iou ---
_AUTO_LABELING_IOU_MODELS = [
    "remote_server",
    "upn",
    "damo_yolo",
    "gold_yolo",
    "yolo_nas",
    "yolov5_obb",
    "yolov5_seg",
    "yolov5_det_track",
    "yolov5",
    "yolov6",
    "yolov7",
    "yolov8_sam2",
    "yolov8_obb",
    "yolov8_pose",
    "yolov8_seg",
    "yolov8_det_track",
    "yolov8_seg_track",
    "yolov8_obb_track",
    "yolov8_pose_track",
    "yolov8",
    "yolov9",
    "yolo11",
    "yolo11_obb",
    "yolo11_seg",
    "yolo11_pose",
    "yolo11_det_track",
    "yolo11_seg_track",
    "yolo11_obb_track",
    "yolo11_pose_track",
    "yolox",
    "yolo12",
    "yoloe",
]


# --- set_auto_labeling_preserve_existing_annotations_state ---
_AUTO_LABELING_PRESERVE_EXISTING_ANNOTATIONS_STATE_MODELS = [
    "remote_server",
    "damo_yolo",
    "gold_yolo",
    "grounding_dino",
    "grounding_dino_api",
    "rtdetr",
    "rtdetrv2",
    "yolo_nas",
    "yolov5_obb",
    "yolov5_seg",
    "yolov5_det_track",
    "yolov5",
    "yolov6",
    "yolov7",
    "yolov8_sam2",
    "yolov8_obb",
    "yolov8_pose",
    "yolov8_seg",
    "yolov8_det_track",
    "yolov8_seg_track",
    "yolov8_obb_track",
    "yolov8_pose_track",
    "yolov8",
    "yolov9",
    "yolov10",
    "yolo11",
    "yolo11_obb",
    "yolo11_seg",
    "yolo11_pose",
    "yolo11_det_track",
    "yolo11_seg_track",
    "yolo11_obb_track",
    "yolo11_pose_track",
    "yolow",
    "yolox",
    "doclayout_yolo",
    "florence2",
    "rfdetr",
    "rfdetr_seg",
    "deimv2",
    "dfine",
    "yolo12",
    "u_rtdetr",
    "yoloe",
    "segment_anything_2_video",
]


# --- set_auto_labeling_prompt ---
_AUTO_LABELING_PROMPT_MODELS = [
    "segment_anything_2_video",
]


# --- on_next_files_changed ---
_ON_NEXT_FILES_CHANGED_MODELS = [
    "segment_anything",
    "segment_anything_2",
    "sam_med2d",
    "sam_hq",
    "yolov5_sam",
    "yolov8_sam2",
    "efficientvit_sam",
    "grounding_sam",
    "grounding_sam2",
    "edge_sam",
    "geco",
]


# --- update_thumbnail_display ---
_THUMBNAIL_RENDER_MODELS = {
    "rmbg": ("x-anylabeling-matting", ".png"),
    "depth_anything": ("x-anylabeling-depth", ".png"),
    "depth_anything_v2": ("x-anylabeling-depth", ".png"),
}
