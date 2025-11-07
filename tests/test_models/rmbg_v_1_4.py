"""
This module provides functionality for removing image backgrounds using ONNX runtime.
"""

import numpy as np
import onnxruntime
from skimage import io
from PIL import Image
import cv2


class ImageBackgroundRemover:
    """A class for removing backgrounds from images using an ONNX model."""

    def __init__(self, model_path: str, providers: list = None):
        """
        Initialize the ImageBackgroundRemover.

        Args:
            model_path (str): Path to the ONNX model file.
            providers (list, optional): List of execution providers for ONNX runtime.
                Defaults to ['CPUExecutionProvider'].
        """
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.ort_session = onnxruntime.InferenceSession(
            model_path, providers=providers
        )
        self.model_input_size = (1024, 1024)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for the ONNX model.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Preprocessed image.
        """
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=2)
        image = cv2.resize(
            image, self.model_input_size[::-1], interpolation=cv2.INTER_LINEAR
        )
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 1.0
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0)

    def postprocess_image(
        self, result: np.ndarray, original_size: tuple
    ) -> np.ndarray:
        """
        Postprocess the model output.

        Args:
            result (np.ndarray): Model output.
            original_size (tuple): Original image size (height, width).

        Returns:
            np.ndarray: Postprocessed image as a numpy array.
        """
        result = cv2.resize(
            np.squeeze(result),
            original_size[::-1],
            interpolation=cv2.INTER_LINEAR,
        )
        max_val, min_val = np.max(result), np.min(result)
        result = (result - min_val) / (max_val - min_val)
        return (result * 255).astype(np.uint8)

    def remove_background(self, image_path: str, output_path: str):
        """
        Remove the background from an image and save the result.

        Args:
            image_path (str): Path to the input image.
            output_path (str): Path to save the output image.
        """
        # Load and preprocess the image
        orig_image = io.imread(image_path)
        orig_size = orig_image.shape[:2]
        preprocessed_image = self.preprocess_image(orig_image)

        # Run inference
        ort_inputs = {
            self.ort_session.get_inputs()[0].name: preprocessed_image
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        result = ort_outs[0]

        # Postprocess the result
        result_image = self.postprocess_image(result[0][0], orig_size)

        # Create the final image with transparent background
        pil_mask = Image.fromarray(result_image)
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGBA")
        pil_mask = pil_mask.convert("L")

        # Create a new image with an alpha channel
        output_image = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        output_image.paste(pil_image, (0, 0), pil_mask)

        # Save the result
        output_image.save(output_path)
        print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    remover = ImageBackgroundRemover("model.onnx")
    remover.remove_background("dog.jpg", "output_no_bg.png")
