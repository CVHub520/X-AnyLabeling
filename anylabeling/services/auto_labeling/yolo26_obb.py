from .__base__.yolo import YOLO


class YOLO26_OBB(YOLO):
    class Meta(YOLO.Meta):
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "toggle_preserve_existing_annotations",
        ]
