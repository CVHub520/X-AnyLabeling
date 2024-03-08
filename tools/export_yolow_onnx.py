import os
import cv2
import time as time
import numpy as np
import onnxruntime as ort

"""
The onnxruntime demo of the YOLO-World
Written by Wei Wang (CVHub)
    Usage:
        1. Download source code from [huggingface/stevengrove/YOLO-World](https://huggingface.co/spaces/stevengrove/YOLO-World)
        2. cd YOLO-World and pip install -r requirements.txt
        3. export PYTHONPATH=/path/to/your/YOLO-World
        4. execute `python app.py --config ${your_custom_config.py} --checkpoint ${your_custom_finetune_model_config.py}` 
        5. Place the current script in this directory
        6. Put the corresponding onnx weight into the specific directory
        7. Run the following command
        ```bash
        python ${export_yolow_onnx.py}
        ```
"""

import onnxruntime as ort
import numpy as np
import cv2


def denormalize_bbox(bbox, input_shape, image_shape):
    """
    Denormalizes bounding box coordinates from input_shape to image_shape.

    Parameters:
    - bbox: Normalized bounding box coordinates [xmin, ymin, xmax, ymax]
    - input_shape: The shape of the input image used during normalization (e.g., [640, 640])
    - image_shape: The shape of the original image (e.g., [height, width])

    Returns:
    - Denormalized bounding box coordinates [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox

    # Denormalize x-coordinates
    denorm_xmin = int(xmin * image_shape[1] / input_shape[1])
    denorm_xmax = int(xmax * image_shape[1] / input_shape[1])

    # Denormalize y-coordinates
    denorm_ymin = int(ymin * image_shape[0] / input_shape[0])
    denorm_ymax = int(ymax * image_shape[0] / input_shape[0])

    denormalized_bbox = [denorm_xmin, denorm_ymin, denorm_xmax, denorm_ymax]

    return denormalized_bbox


def preprocess_image(image_path, input_shape):
    im0 = cv2.imread(image_path)
    image_shape = im0.shape[:2]
    image = cv2.resize(
        im0, input_shape
    )  # Resize to the input dimension expected by the YOLO model
    image = image.astype(np.float32) / 255.0  # Normalize the image
    image = np.transpose(
        image, (2, 0, 1)
    )  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    return image, image_shape, im0


def inference(session, input_name, image):
    outputs = session.run(None, {input_name: image})
    num_objs, bboxes, scores, class_ids = [out[0] for out in outputs]
    return num_objs, bboxes, scores, class_ids


def postprocess_results(
    output_image,
    scores,
    class_ids,
    bbox,
    input_shape,
    image_shape,
    score_threshold,
):
    for i, score in enumerate(scores):
        if score > score_threshold and (class_ids[i] != -1):
            bbox[i] = denormalize_bbox(bbox[i], input_shape, image_shape)
            x_min, y_min, x_max, y_max = bbox[i]
            start_point = (int(x_min), int(y_min))
            end_point = (int(x_max), int(y_max))
            color = (0, 255, 0)
            cv2.rectangle(output_image, start_point, end_point, color, 2)
            label = f"{class_ids[i]}: {score:.2f}"
            cv2.putText(
                output_image,
                label,
                (int(x_min), int(y_min) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    return output_image


def forward(image_path, onnx_path, score_threshold):
    session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape[-2:]

    blob, image_shape, im0 = preprocess_image(image_path, input_shape)
    _, bboxes, scores, class_ids = inference(session, input_name, blob)
    output_image = postprocess_results(
        im0,
        scores,
        class_ids,
        bboxes,
        input_shape,
        image_shape,
        score_threshold,
    )

    return output_image


def main():
    score_threshold = 0.05
    image_path = "/path/to/image"
    model_path = "/path/to/model"

    result_image = forward(image_path, model_path, score_threshold)

    try:
        cv2.imshow("Detected Objects", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        cv2.imwrite("/path/to/save", result_image)


if __name__ == "__main__":
    main()
