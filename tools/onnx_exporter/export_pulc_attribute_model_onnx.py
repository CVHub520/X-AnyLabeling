import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort


"""
The ONNX Export of the PULC Attribute Model
Written by Wei Wang (CVHub)

    Export:
        1. git clone https://github.com/PaddlePaddle/PaddleClas.git
        2. cd PaddleClas and follow the official tutorial to install paddle and paddlecls env.
        3. Download the weight file and refer to the following running script:
        ```bash
            # Vehicle
            paddle2onnx \
                --model_dir=./models/vehicle_attribute_infer/ \
                --model_filename=inference.pdmodel \
                --params_filename=inference.pdiparams \
                --save_file=./models/vehicle_attribute_infer/inference.onnx \
                --opset_version 12 \
                --deploy_backend onnxruntime \
                --enable_auto_update_opset True \
                --enable_onnx_checker True
            python -m paddle2onnx.optimize \
                --input_model ./models/vehicle_attribute_infer/inference.onnx \
                --output_model ./models/vehicle_attribute_infer/pulc_vehicle_attribute.onnx
            -------------------------------------------------------------------------------
            # Person
            paddle2onnx --model_dir=./models/person_attribute_infer/ \
                --model_filename=inference.pdmodel \
                --params_filename=inference.pdiparams \
                --save_file=./models/person_attribute_infer/inference.onnx \
                --opset_version 12 \
                --enable_auto_update_opset True \
                --enable_onnx_checker True
            python -m paddle2onnx.optimize \
                --input_model ./models/person_attribute_infer/inference.onnx \
                --output_model ./models/person_attribute_infer/pulc_person_attribute.onnx
        ```
    Usage:
        ```bash
            python models/demo.py \
                dataset/pulc_demo_imgs/vehicle_attribute/0002_c002_00030670_0.jpg \
                ./models/vehicle_attribute_infer/pulc_vehicle_attribute.onnx \
                --task vehicle
            python models/demo.py \
                dataset/pulc_demo_imgs/person_attribute/090004.jpg \
                ./models/person_attribute_infer/pulc_person_attribute.onnx \
                --task person
        ```
"""


class OnnxBaseModel:
    def __init__(self, model_path, device_type: str = "cpu"):
        self.sess_opts = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(
                os.environ["OMP_NUM_THREADS"]
            )

        self.providers = ["CPUExecutionProvider"]
        if device_type.lower() != "cpu":
            self.providers = ["CUDAExecutionProvider"]

        self.ort_session = ort.InferenceSession(
            model_path,
            providers=self.providers,
            sess_options=self.sess_opts,
        )

    def get_ort_inference(
        self, blob, inputs=None, extract=True, squeeze=False
    ):
        if inputs is None:
            inputs = self.get_input_name()
            outs = self.ort_session.run(None, {inputs: blob})
        else:
            outs = self.ort_session.run(None, inputs)
        if extract:
            outs = outs[0]
        if squeeze:
            outs = outs.squeeze(axis=0)
        return outs

    def get_input_name(self):
        return self.ort_session.get_inputs()[0].name

    def get_input_shape(self):
        return self.ort_session.get_inputs()[0].shape

    def get_output_name(self):
        return [out.name for out in self.ort_session.get_outputs()]


class PULC_Attribute(object):
    def __init__(self, model_abs_path, task="vehicle"):
        self.vehicle_attributes = {
            "Color": [
                [
                    "yellow",
                    "orange",
                    "green",
                    "gray",
                    "red",
                    "blue",
                    "white",
                    "golden",
                    "brown",
                    "black",
                ],
                -1,
            ],
            "Type": [
                [
                    "sedan",
                    "suv",
                    "van",
                    "hatchback",
                    "mpv",
                    "pickup",
                    "bus",
                    "truck",
                    "estate",
                ],
                -1,
            ],
        }
        self.person_attributes = {
            "Hat": [["Yes", "False"], 0.5],
            "Glasses": [["Yes", "No"], 0.3],
            "Sleeve": [["ShortSleeve", "LongSleeve"], -1],
            "UpperStride": [["Yes", "No"], 0.5],
            "UpperLogo": [["Yes", "No"], 0.5],
            "UpperPlaid": [["Yes", "No"], 0.5],
            "UpperSplice": [["Yes", "No"], 0.5],
            "LowerStripe": [["Yes", "No"], 0.5],
            "LowerPattern": [["Yes", "No"], 0.5],
            "LongCoat": [["Yes", "No"], 0.5],
            "Trousers": [["Yes", "No"], 0.5],
            "Shorts": [["Yes", "No"], 0.5],
            "Skirt&Dress": [["Yes", "No"], 0.5],
            "Shoe": [["Boots", "No boots"], 0.5],
            "HandBag&Dress": [["Yes", "No"], 0.5],
            "ShoulderBag&Dress": [["Yes", "No"], 0.5],
            "Backpack": [["Yes", "No"], 0.5],
            "HoldObjectsInFront": [["Yes", "No"], 0.6],
            "Age": [["AgeLess18", "Age18-60", "AgeOver60"], -1],
            "Gender": [["Female", "Male"], 0.5],
            "Direction": [["Front", "Side", "Back"], -1],
        }
        if task == "vehicle":
            self.attributes = self.vehicle_attributes
        elif task == "person":
            self.attributes = self.person_attributes
        else:
            raise ValueError(f"Invalid task mode: {task}!")

        self.net = OnnxBaseModel(model_abs_path, device_type="cpu")
        self.input_shape = self.net.get_input_shape()[-2:][::-1]

    def preprocess(self, input_image):
        """
        Post-processes the network's output.
        """
        image = cv2.resize(input_image, self.input_shape, interpolation=1)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = np.array(mean).reshape((1, 1, 3)).astype("float32")
        std = np.array(std).reshape((1, 1, 3)).astype("float32")
        image = (
            image.astype("float32") * np.float32(1.0 / 255.0) - mean
        ) / std
        image = image.transpose(2, 0, 1).astype("float32")
        image = np.expand_dims(image, axis=0)
        return image

    def postprocess(self, outs):
        """
        Predict shapes from image
        """
        outs = outs.tolist()

        interval = 0
        results = {}
        for property, infos in self.attributes.items():
            options, threshold = infos
            if threshold == -1:
                num_classes = len(options)
                current_class = outs[interval : interval + num_classes]
                current_index = np.argmax(current_class)
                results[property] = options[current_index]
                interval += num_classes
            elif 0.0 <= threshold <= 1.0:
                current_score = outs[interval]
                current_class = (
                    options[0] if current_score > threshold else options[1]
                )
                results[property] = current_class
                interval += 1

        return results

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        blob = self.preprocess(image)
        outputs = self.net.get_ort_inference(blob, squeeze=True)
        results = self.postprocess(outputs)

        return results

    def unload(self):
        del self.net


def main():
    parser = argparse.ArgumentParser(
        description="PaddleCls PULC Inference Demo"
    )
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument(
        "model_path",
        type=str,
        default="model.onnx",
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="vehicle",
        choices=["vehicle", "person"],
        help="Task mode",
    )
    args = parser.parse_args()

    inference_model = PULC_Attribute(args.model_path, task=args.task)
    img = cv2.imread(args.image_path)
    img = img[:, :, ::-1]
    result = inference_model.predict_shapes(img)

    print("The final inference results:")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
