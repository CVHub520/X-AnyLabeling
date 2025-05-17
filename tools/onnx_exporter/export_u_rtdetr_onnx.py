import os
import cv2
import random
import numpy as np
import onnxruntime as ort

"""
The onnxruntime demo of the Ultralytics RT-DETR
Written by Wei Wang (CVHub)
    Usage:
        1. git clone https://github.com/ultralytics/ultralytics
        2. cd ultralytics and pip install -r requirements.txt
        3. export PYTHONPATH=/path/to/your/ultralytics
        4. Place the current script in this directory
        5. Download the corresponding weights
        6. Export the model to onnx format:
        ```bash
        yolo export model=rtdetr-l/x.pt format=onnx opset=16
        ```
        7. Modified the values and run this script
        ```bash
        python ${export_u_rtdetr_onnx.py}
        ```
"""

random.seed(10086)
CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
CLASS_COLORS = [
    [random.randint(0, 255) for _ in range(3)] for _ in range(len(CLASS_NAMES))
]


def read_labels(label_path):
    """Read labels from a file."""
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            labels.append(line.strip())
    return labels


def preprocess(image, input_shape):
    """Preprocess the input image for RTDETR model."""
    # Get original image dimensions
    image_height, image_width = image.shape[:2]

    # Compute rescale ratio
    ratio_height = input_shape[0] / image_height
    ratio_width = input_shape[1] / image_width

    # Resize the image
    resized_image = cv2.resize(
        image,
        (0, 0),
        fx=ratio_width,
        fy=ratio_height,
        interpolation=cv2.INTER_LINEAR,
    )

    # Convert BGR to RGB
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Normalize image to [0, 1]
    resized_image_norm = resized_image_rgb.astype(np.float32) / 255.0

    # Convert HWC to CHW (height, width, channels) -> (channels, height, width)
    blob = resized_image_norm.transpose(2, 0, 1)

    # Add batch dimension: NCHW
    blob = np.expand_dims(blob, axis=0)

    # Make sure it's contiguous in memory
    blob = np.ascontiguousarray(blob)

    return blob


def bbox_cxcywh_to_xyxy(boxes):
    """Convert bounding boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format."""
    xyxy_boxes = []
    for box in boxes:
        x1 = box[0] - box[2] / 2.0
        y1 = box[1] - box[3] / 2.0
        x2 = box[0] + box[2] / 2.0
        y2 = box[1] + box[3] / 2.0
        xyxy_boxes.append([x1, y1, x2, y2])
    return xyxy_boxes


def is_normalized(values):
    """Check if the values are already normalized (between 0 and 1)."""
    for row in values:
        for val in row:
            if val <= 0 or val >= 1:
                return False
    return True


def normalize_scores(scores):
    """Apply sigmoid normalization to scores if needed."""
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            scores[i][j] = 1 / (1 + np.exp(-scores[i][j]))


def rescale_boxes(boxes, image_shape, input_shape):
    """Rescale boxes from model input shape to original image shape."""
    image_height, image_width = image_shape

    # Extract boxes
    x1 = np.array([box[0] for box in boxes])
    y1 = np.array([box[1] for box in boxes])
    x2 = np.array([box[2] for box in boxes])
    y2 = np.array([box[3] for box in boxes])

    # Rescale from normalized coordinates to pixel coordinates
    x1 = np.floor(np.clip(x1 * image_width, 0, image_width - 1))
    y1 = np.floor(np.clip(y1 * image_height, 0, image_height - 1))
    x2 = np.ceil(np.clip(x2 * image_width, 0, image_width - 1))
    y2 = np.ceil(np.clip(y2 * image_height, 0, image_height - 1))

    # Create new boxes
    new_boxes = []
    for i in range(len(boxes)):
        new_boxes.append([x1[i], y1[i], x2[i], y2[i]])

    return new_boxes


def draw_boxes(image, labels, scores, boxes, class_names):
    """Draw bounding boxes and labels on the image."""
    for i, (label, score, box) in enumerate(zip(labels, scores, boxes)):
        color = CLASS_COLORS[label]
        x1, y1, x2, y2 = map(int, box)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label_text = f"{class_names[label]}: {score:.2f}"

        # Get text size
        text_size = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )[0]

        # Fill background rectangle for text
        cv2.rectangle(
            image,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            color,
            -1,
        )

        # Put text
        cv2.putText(
            image,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return image


def process_rtdetr(
    image, model_session, input_shape, conf_threshold, class_names
):
    """Process image with RT-DETR model and return detection results."""
    # Save original image dimensions
    image_shape = image.shape[:2]

    # Preprocess image
    blob = preprocess(image, input_shape)

    # Run inference
    outputs = model_session.run(None, {"images": blob})[0][0]

    # Extract boxes and scores
    num_boxes = outputs.shape[0]
    num_classes = len(class_names)

    # Parse boxes and scores (first 4 elements are box coordinates, rest are class scores)
    boxes = []
    scores = []
    for i in range(num_boxes):
        box = outputs[i, :4]
        score = outputs[i, 4 : 4 + num_classes]
        boxes.append(box)
        scores.append(score)

    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
    xyxy_boxes = bbox_cxcywh_to_xyxy(boxes)

    # Normalize scores if needed
    if not is_normalized(scores):
        normalize_scores(scores)

    # Get max scores and corresponding class indices
    max_scores = np.max(scores, axis=1)
    class_indices = np.argmax(scores, axis=1)

    # Filter detections based on confidence threshold
    mask = max_scores > conf_threshold
    filtered_boxes = [xyxy_boxes[i] for i in range(len(xyxy_boxes)) if mask[i]]
    filtered_scores = [
        max_scores[i] for i in range(len(max_scores)) if mask[i]
    ]
    filtered_class_indices = [
        class_indices[i] for i in range(len(class_indices)) if mask[i]
    ]

    # Rescale boxes to original image size
    filtered_boxes = rescale_boxes(filtered_boxes, image_shape, input_shape)
    return filtered_class_indices, filtered_scores, filtered_boxes


def process_image(
    image_path, model_path, save_path, conf_threshold=0.45, use_cuda=False
):
    """Process a single image and save the result."""
    # Initialize ONNX Runtime
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(model_path, providers=providers)

    # Get model input shape from the model's first input
    input_shape = session.get_inputs()[0].shape
    input_height, input_width = input_shape[2:4]

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Process image with RT-DETR
    labels, scores, boxes = process_rtdetr(
        image,
        session,
        (input_height, input_width),
        conf_threshold,
        CLASS_NAMES,
    )

    # Draw boxes on image
    result_image = draw_boxes(image.copy(), labels, scores, boxes, CLASS_NAMES)

    # Save result
    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        cv2.imwrite(save_path, result_image)
        print(f"Result saved to {save_path}")

    return result_image, labels, scores, boxes


def process_video(
    video_path, model_path, save_path=None, conf_threshold=0.45, use_cuda=False
):
    """Process a video and save or display the result."""
    # Initialize ONNX Runtime
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(model_path, providers=providers)

    # Get model input shape from the model's first input
    input_shape = session.get_inputs()[0].shape
    input_height, input_width = input_shape[2:4]

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer if saving output
    writer = None
    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Start timer
        start_time = cv2.getTickCount()

        # Process frame with RT-DETR
        labels, scores, boxes = process_rtdetr(
            frame,
            session,
            (input_height, input_width),
            conf_threshold,
            CLASS_NAMES,
        )

        # Draw boxes on frame
        result_frame = draw_boxes(
            frame.copy(), labels, scores, boxes, CLASS_NAMES
        )

        # Calculate FPS
        fps_frame = cv2.getTickFrequency() / (cv2.getTickCount() - start_time)

        # Put FPS on frame
        cv2.putText(
            result_frame,
            f"FPS: {fps_frame:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Write frame to output video if saving
        if writer:
            writer.write(result_frame)

        # Display frame
        cv2.imshow("RT-DETR", result_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    # Configuration parameters
    conf_threshold = 0.45
    use_cuda = False
    input_shape = (640, 640)

    # File paths
    image_path = "/path/to/image.jpg"
    video_path = "./path/to/video.mp4"
    save_path = "./dist/output.jpg"
    model_path = "/path/to/rtdetr-l/x.onnx"

    # Choose processing mode
    process_mode = "image"  # "image" or "video"

    if process_mode == "image":
        # Process image
        process_image(
            image_path, model_path, save_path, conf_threshold, use_cuda
        )
    elif process_mode == "video":
        # Process video
        process_video(
            video_path, model_path, save_path, conf_threshold, use_cuda
        )
    else:
        print(f"Invalid process mode: {process_mode}")


if __name__ == "__main__":
    main()
