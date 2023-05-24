class AutoLabelingResult:
    def __init__(self, shapes, replace=True):
        """Initialize AutoLabelingResult

        Args:
            shapes (List[Shape]): List of shapes to add to the canvas.
            replace (bool, optional): Replaces all current shapes with
            new shapes. Defaults to True.
        """

        self.shapes = shapes
        self.replace = replace


class AutoLabelingMode:
    OBJECT = "AUTOLABEL_OBJECT"
    ADD = "AUTOLABEL_ADD"
    REMOVE = "AUTOLABEL_REMOVE"
    POINT = "point"
    RECTANGLE = "rectangle"

    def __init__(self, edit_mode, shape_type):
        """Initialize AutoLabelingMode

        Args:
            edit_mode (str): AUTOLABEL_ADD / AUTOLABEL_REMOVE
            shape_type (str): point / rectangle
        """

        self.edit_mode = edit_mode
        self.shape_type = shape_type

    @staticmethod
    def get_default_mode():
        """Get default mode"""
        return AutoLabelingMode(AutoLabelingMode.ADD, AutoLabelingMode.POINT)

    # Compare 2 instances of AutoLabelingMode
    def __eq__(self, other):
        if not isinstance(other, AutoLabelingMode):
            return False
        return (
            self.edit_mode == other.edit_mode
            and self.shape_type == other.shape_type
        )


AutoLabelingMode.NONE = AutoLabelingMode(None, None)
