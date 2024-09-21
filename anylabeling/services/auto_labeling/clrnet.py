import os
import cv2
import numpy as np
import onnxruntime as ort

from scipy.interpolate import InterpolatedUnivariateSpline

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult


class CLRNet(Model):
    """Lane detection model using CLRNet (CVPR 2022)"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "input_width",
            "input_height",
            "image_width",
            "image_height",
            "nms_threshold",
            "conf_threshold",
            "n_offsets",
            "max_lanes",
            "cut_height",
        ]
        widgets = ["button_run", "toggle_preserve_existing_annotations"]
        output_modes = {
            "line": QCoreApplication.translate("Model", "Line"),
        }
        default_output_mode = "line"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize CLRNet model."
                )
            )

        self.sess_opts = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(
                os.environ["OMP_NUM_THREADS"]
            )
        self.providers = ["CPUExecutionProvider"]
        if __preferred_device__ == "GPU":
            self.providers = ["CUDAExecutionProvider"]

        self.net = ort.InferenceSession(
            model_abs_path,
            providers=self.providers,
            sess_options=self.sess_opts,
        )
        self.n_offsets = self.config["n_offsets"]
        self.n_strips = self.n_offsets - 1
        self.max_lanes = self.config["max_lanes"]
        self.nms_thres = self.config["nms_threshold"]
        self.conf_thres = self.config["conf_threshold"]
        self.cut_height = self.config["cut_height"]
        self.prior_ys = np.linspace(1, 0, self.n_offsets)
        self.input_size = (
            self.config["input_width"],
            self.config["input_height"],
        )
        self.image_size = (
            self.config["image_width"],
            self.config["image_height"],
        )
        self.replace = True

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def pre_process(self, input_image, net):
        """
        Pre-process the input RGB image before feeding it to the network.
        """
        self.ori_image_height, self.ori_image_width = input_image.shape[:2]
        image = cv2.resize(input_image, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image[self.cut_height :, :, :]
        image = cv2.resize(image, self.input_size, cv2.INTER_CUBIC)
        image = image.transpose((2, 0, 1))  # HWC to CHW
        image = image.astype("float32")
        image /= 255  # 0 - 255 to 0.0 - 1.0
        if len(image.shape) == 3:
            image = image[None]

        inputs = net.get_inputs()[0].name
        outputs = net.run(None, {inputs: image})[0]
        return outputs

    def post_process(self, outputs):
        """
        Post-process the network's output
        """
        results = []
        for output in outputs:
            scores = self.softmax(output[:, :2], 1)[:, 1]
            keep_inds = scores >= self.conf_thres
            filter_outputs = output[keep_inds]
            scores = scores[keep_inds]
            if filter_outputs.shape[0] == 0:
                continue

            tmp_outputs = filter_outputs
            tmp_outputs = np.concatenate(
                [tmp_outputs[..., :4], tmp_outputs[..., 5:]], axis=-1
            )
            tmp_outputs[..., 4] = tmp_outputs[..., 4] * self.n_strips
            tmp_outputs[..., 5:] = tmp_outputs[..., 5:] * (
                self.image_size[0] - 1
            )
            nms_keep_inds = self.numpy_land_nms(tmp_outputs, scores)
            nms_predictions = filter_outputs[nms_keep_inds]
            if nms_predictions.shape[0] == 0:
                continue

            nms_predictions[:, 5] = np.round(
                nms_predictions[:, 5] * self.n_strips
            )
            out_predictions = self.convert_outputs(nms_predictions)
            results = out_predictions

        lane_points = []
        for lane in results:
            points = []
            for _x, _y in lane:
                if _x <= 0 or _y <= 0:
                    continue
                _x, _y = int(_x), int(_y)
                points.append((_x, _y))
            lane_points.append(points)
        lane_points.sort(key=lambda points: points[0][0])

        output_infos = []
        for land_point in lane_points:
            info = {
                "points": [],
            }
            for point in land_point:
                ori_x, ori_y = point
                new_x = int(
                    ori_x * (self.ori_image_width / self.image_size[0])
                )
                new_y = int(
                    ori_y * (self.ori_image_height / self.image_size[1])
                )
                info["points"].append([new_x, new_y])
            output_infos.append(info)

        return output_infos

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        outputs = self.pre_process(image, self.net)
        infos = self.post_process(outputs)

        shapes = []
        for i, info in enumerate(infos):
            label = "lane" + str(i + 1)
            shape = Shape(label=label, shape_type="line", flags={})
            start_point, end_point = info["points"][0], info["points"][-1]
            shape = Shape(label=label, shape_type="line", flags={})
            shape.add_point(QtCore.QPointF(start_point[0], start_point[1]))
            shape.add_point(QtCore.QPointF(end_point[0], end_point[1]))
            shapes.append(shape)
        result = AutoLabelingResult(shapes, replace=self.replace)

        return result

    def softmax(self, x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def land_iou(self, a, b):
        start_a = int(a[2] * (self.n_offsets - 1) + 0.5)
        start_b = int(b[2] * (self.n_offsets - 1) + 0.5)
        start = max(start_a, start_b)
        end_a = int(start_a + a[4] - 1 + 0.5 - ((a[4] - 1) < 0))
        end_b = int(start_b + b[4] - 1 + 0.5 - ((b[4] - 1) < 0))
        end = min(min(end_a, end_b), self.n_offsets - 1)

        if end < start:
            return False

        dist = np.sum(
            np.abs(a[5 + start : 5 + end + 1] - b[5 + start : 5 + end + 1])
        )
        return dist < self.nms_thres * (end - start + 1)

    def numpy_land_nms(self, lanes, scores):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        keep_lanes = []
        while sorted_indices.size > 0 and len(keep_lanes) < self.max_lanes:
            # Pick the last box
            land_id = sorted_indices[0]
            keep_lanes.append(land_id)

            # Compute score of the picked lands with the rest
            if sorted_indices.size > 1:
                rest_flags = []
                rest_indices = sorted_indices[1:]
                for i in rest_indices:
                    rest_flags.append(self.land_iou(lanes[land_id], lanes[i]))
                # Remove lands with land_iou over the threshold
                rest_flags = np.array(rest_flags)
                sorted_indices = rest_indices[~rest_flags]
            else:
                break
        return keep_lanes

    def convert_outputs(self, nms_predictions):
        lanes = []
        for lane in nms_predictions:
            lane_xs = lane[6:]
            start_point = min(
                max(0, int(round(lane[2].item() * self.n_strips))),
                self.n_strips,
            )
            lane_length = int(round(lane[5].item()))
            end_point = start_point + lane_length - 1
            end_point = min(end_point, len(self.prior_ys) - 1)
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~(
                (
                    (
                        (lane_xs[:start_point] >= 0.0)
                        & (lane_xs[:start_point] <= 1.0)
                    )[::-1].cumprod()[::-1]
                ).astype(bool)
            )
            lane_xs[end_point + 1 :] = -2
            lane_xs[:start_point][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = np.double(lane_xs)
            lane_xs = np.flip(lane_xs, axis=0)
            lane_ys = np.flip(lane_ys, axis=0)
            lane_ys = (
                lane_ys * (self.image_size[1] - self.cut_height)
                + self.cut_height
            ) / self.image_size[1]
            if len(lane_xs) <= 1:
                continue
            points = np.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), axis=1
            ).squeeze(2)
            spline = InterpolatedUnivariateSpline(
                points[:, 1], points[:, 0], k=min(3, len(points) - 1)
            )
            invalid_value = -2.0
            min_y = points[:, 1].min() - 0.01
            max_y = points[:, 1].max() + 0.01
            # ys
            sample_y = range(710, 150, -10)
            ys = np.array(sample_y) / float(self.image_size[1])
            # xs
            xs = spline(ys)
            xs[(ys < min_y) | (ys > max_y)] = invalid_value
            # mask
            valid_mask = (xs >= 0) & (xs < 1)
            lane_xs = xs[valid_mask] * self.image_size[0]
            lane_ys = ys[valid_mask] * self.image_size[1]
            lane = np.concatenate(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), axis=1
            )
            lanes.append(lane)
        return lanes

    def unload(self):
        del self.net
