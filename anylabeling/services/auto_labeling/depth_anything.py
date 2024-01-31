import logging
import os

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel


class Resize(object):
    """Resize sample to given size (width, height)."""

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(
                int
            )

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(
                int
            )

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(
                f"resize_method {self.__resize_method} not implemented"
            )

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "semseg_mask" in sample:
                sample["semseg_mask"] = cv2.resize(
                    sample["semseg_mask"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                # sample["mask"] = sample["mask"].astype(bool)

        # print(sample['image'].shape, sample['depth'].shape)
        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std."""

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])

        return sample


class DepthAnything(Model):
    """DepthAnything demo"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_name = self.config["type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_name} model.",
                )
            )
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.input_shape = self.net.get_input_shape()[-2:]

    def preprocess(self, input_image):
        """
        Pre-processes the input image before feeding it to the network.

        Args:
            input_image (numpy.ndarray): The input image to be processed.

        Returns:
            numpy.ndarray: The pre-processed output.
        """
        orig_shape = input_image.shape[:2]
        image = input_image / 255.0

        # Apply Resize transformation manually
        resize_transform = Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )
        image = resize_transform({"image": image})["image"]

        # Apply NormalizeImage transformation manually
        normalize_transform = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize_transform({"image": image})["image"]

        # Apply PrepareForNet transformation manually
        prepare_for_net_transform = PrepareForNet()
        image = prepare_for_net_transform({"image": image})["image"]
        image = image[None]  # B, C, H, W

        return image, orig_shape

    def postprocess(self, outputs, orig_shape):
        """
        Post-processes the network's output.

        Args:
            outputs (numpy.ndarray): The output from the network.

        Returns:
            depth_color: xxx
        """
        orig_h, orig_w = orig_shape
        depth = cv2.resize(outputs[0, 0], (orig_w, orig_h))
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        return depth_color

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """
        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        blob, orig_shape = self.preprocess(image)
        output = self.net.get_ort_inference(blob)
        depth = self.postprocess(output, orig_shape)

        image_dir_path = os.path.dirname(image_path)
        save_path = os.path.join(image_dir_path, "..", "depth")
        save_path = os.path.realpath(save_path)
        os.makedirs(save_path, exist_ok=True)
        image_file_name = os.path.basename(image_path)
        save_name = os.path.splitext(image_file_name)[0] + ".png"
        save_file = os.path.join(save_path, save_name)
        cv2.imwrite(save_file, depth)

        return AutoLabelingResult([], replace=False)

    def unload(self):
        del self.net
