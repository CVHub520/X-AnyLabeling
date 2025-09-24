import os
import cv2
import numpy as np
import onnxruntime as ort

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .utils.ppocr_utils.text_system import TextSystem
from ...views.labeling.utils.general import is_possible_rectangle


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class PPOCRv4(Model):
    """PaddlePaddle OCR-v4"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "det_model_path",
            "rec_model_path",
            "cls_model_path",
            "use_angle_cls",
        ]
        widgets = ["button_run", "button_skip_detection"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def load_model(self, model_name):
        model_abs_path = self.get_model_abs_path(self.config, model_name)
        model_task = os.path.splitext(
            os.path.basename(self.config[model_name])
        )[0]
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_task} model.",
                )
            )

        self.sess_opts = ort.SessionOptions()
        self.sess_opts.log_severity_level = 3
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(
                os.environ["OMP_NUM_THREADS"]
            )
        self.providers = ["CPUExecutionProvider"]

        if __preferred_device__ == "GPU":
            self.providers = ["CUDAExecutionProvider"]
        net = ort.InferenceSession(
            model_abs_path,
            providers=self.providers,
            sess_options=self.sess_opts,
        )
        return net

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        self.det_net = self.load_model("det_model_path")
        self.rec_net = self.load_model("rec_model_path")
        self.cls_net = self.load_model("cls_model_path")
        self.drop_score = self.config.get("drop_score", 0.5)
        self.use_angle_cls = self.config["use_angle_cls"]
        self.current_dir = os.path.dirname(__file__)
        self.lang = self.config.get("lang", "ch")
        if self.lang == "ch":
            self.rec_char_dict = "ppocr_keys_v1.txt"
        elif self.lang == "japan":
            self.rec_char_dict = "japan_dict.txt"
        elif self.lang == "ppocrv5_dict":
            self.rec_char_dict = "ppocrv5_dict.txt"

        self.args = self.parse_args()
        self.text_sys = TextSystem(self.args)

    def parse_args(self):
        args = Args(
            use_onnx=True,
            # params for prediction engine
            use_gpu=True,
            use_xpu=False,
            use_npu=False,
            ir_optim=True,
            use_tensorrt=False,
            min_subgraph_size=15,
            precision="fp32",
            gpu_mem=500,
            gpu_id=0,
            # params for text detector
            page_num=0,
            det_algorithm="DB",
            det_model=self.det_net,
            det_limit_side_len=960,
            det_limit_type="max",
            det_box_type="quad",
            # DB parmas
            det_db_thresh=0.3,
            det_db_box_thresh=0.6,
            det_db_unclip_ratio=1.5,
            max_batch_size=10,
            use_dilation=False,
            det_db_score_mode="fast",
            # EAST parmas
            det_east_score_thresh=0.8,
            det_east_cover_thresh=0.1,
            det_east_nms_thresh=0.2,
            # SAST parmas
            det_sast_score_thresh=0.5,
            det_sast_nms_thresh=0.2,
            # PSE parmas
            det_pse_thresh=0,
            det_pse_box_thresh=0.85,
            det_pse_min_area=16,
            det_pse_scale=1,
            # FCE parmas
            scales=[8, 16, 32],
            alpha=1.0,
            beta=1.0,
            fourier_degree=5,
            # params for text recognizer
            rec_algorithm="SVTR_LCNet",
            rec_model=self.rec_net,
            rec_image_inverse=True,
            rec_image_shape="3, 48, 320",
            rec_batch_num=6,
            max_text_length=25,
            rec_char_dict_path=os.path.join(
                self.current_dir, f"configs/ppocr/{self.rec_char_dict}"
            ),
            use_space_char=True,
            drop_score=self.drop_score,
            # params for e2e
            e2e_algorithm="PGNet",
            e2e_model_dir="",
            e2e_limit_side_len=768,
            e2e_limit_type="max",
            # PGNet parmas
            e2e_pgnet_score_thresh=0.5,
            e2e_char_dict_path=os.path.join(
                self.current_dir, "configs/ppocr/ppocr_ic15_dict.txt"
            ),
            e2e_pgnet_valid_set="totaltext",
            e2e_pgnet_mode="fast",
            # params for text classifier
            use_angle_cls=self.use_angle_cls,
            cls_model=self.cls_net,
            cls_image_shape="3, 48, 192",
            label_list=["0", "180"],
            cls_batch_num=6,
            cls_thresh=0.9,
            enable_mkldnn=False,
            cpu_threads=10,
            use_pdserving=False,
            warmup=False,
            # SR parmas
            sr_model_dir="",
            sr_image_shape="3, 32, 128",
            sr_batch_num=1,
        )
        return args

    def predict_shapes(self, image, image_path=None, existing_shapes=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        selected_shapes, unselected_shapes = [], []
        if existing_shapes is not None:
            dt_boxes = []
            for shape in existing_shapes:
                if shape.selected:
                    points = [
                        [int(point.x()), int(point.y())]
                        for point in shape.points
                    ]
                    dt_boxes.append(points)
                    selected_shapes.append(shape)
                else:
                    unselected_shapes.append(shape)

            if not dt_boxes:
                dt_boxes = None
        else:
            dt_boxes = None

        dt_boxes, rec_res, scores, sort_indices = self.text_sys(
            image, dt_boxes=dt_boxes
        )

        if existing_shapes is not None:
            shapes = []
            for i in range(len(selected_shapes)):
                ori_index = sort_indices[i]
                shape = selected_shapes[ori_index]
                updated_shape = Shape(
                    label=shape.label,
                    score=shape.score,
                    shape_type=shape.shape_type,
                    group_id=shape.group_id,
                    description=rec_res[i][0],
                    difficult=shape.difficult,
                    flags=shape.flags,
                    attributes=shape.attributes,
                )
                for point in shape.points:
                    updated_shape.add_point(point)
                if hasattr(shape, "closed"):
                    updated_shape.closed = shape.closed
                shapes.append(updated_shape)

            shapes.extend(unselected_shapes)
            result = AutoLabelingResult(shapes, replace=True)
            return result

        results = [
            {
                "description": rec_res[i][0],
                "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
                "score": float(scores[i]),
            }
            for i in range(len(dt_boxes))
        ]

        shapes = []
        for i, res in enumerate(results):
            score = res["score"]
            points = res["points"]
            description = res["description"]
            shape_type = (
                "rectangle" if is_possible_rectangle(points) else "polygon"
            )
            shape = Shape(
                label="text",
                score=score,
                shape_type=shape_type,
                group_id=int(i),
                description=description,
            )
            if shape_type == "rectangle":
                pt1, pt2, pt3, pt4 = points
                pt2 = [pt3[0], pt1[1]]
                pt4 = [pt1[0], pt3[1]]
                shape.add_point(QtCore.QPointF(*pt1))
                shape.add_point(QtCore.QPointF(*pt2))
                shape.add_point(QtCore.QPointF(*pt3))
                shape.add_point(QtCore.QPointF(*pt4))
            elif shape_type == "polygon":
                for point in points:
                    shape.add_point(QtCore.QPointF(*point))
                shape.add_point(QtCore.QPointF(*points[0]))
                shape.closed = True
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.det_net
        del self.rec_net
        del self.cls_net
        del self.text_sys
