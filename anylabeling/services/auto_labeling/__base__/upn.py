import copy
from typing import Dict, List, Union

import numpy as np
import torch
from mmengine import Config
from PIL import Image
from torchvision.ops import nms

import chatrex.upn.transforms.transform as T
from chatrex.upn import build_architecture
from chatrex.upn.models.module import nested_tensor_from_tensor_list


def build_model(ckpt_path: str):
    config_path = (
        "anylabeling/services/auto_labeling/configs/chatrex/upn_large.py"
    )
    model_cfg = Config.fromfile(config_path).model
    model = build_architecture(model_cfg)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    return model


class UPNWrapper:
    """A wrapper class for the UPN model.

    Args:
        ckpt_path (str): The path to the model checkpoint
        config_path (str): The path to the model config
        device (str): The device to use for inference, e.g. "cuda" or "cpu"
    """

    def __init__(self, ckpt_path: str, device: str):
        self.model = build_model(ckpt_path)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def inference(
        self,
        image: List[Union[str, Image.Image]],
        prompt_type: str,
    ):
        """Single image prediction.

        Args:
            image List[Union[str, Image.Image]]: A list of image path or
                PIL.Image.Image object.
            prompt_type (str): The type of prompt to use for the prediction. Choice in
                ['fine_grained_prompt', 'coarse_grained_prompt'].

        Returns:
           Dict: Return dict in format:
                {
                    "original_xyxy_boxes": (np.ndarray): Original prediction boxes in shape (batch_size, 900, 4),
                    "scores": (np.ndarray): Score in shape (batch_size, N)
                }
        """
        if not isinstance(image, list):
            image = [image]
        input_images, image_sizes = self.construct_input(image)
        outputs = self._inference(input_images, prompt_type)
        post_processed_outputs = self.postprocess(outputs, image_sizes)
        return post_processed_outputs

    def _inference(self, input_images: List[torch.Tensor], prompt_type: str):
        """Inference for T-Rex2

        Args:
            input_images (List[torch.Tensor]): Transformed Image

        Retunrs:
            (Dict): Return dict with keys:
                - query_features: (torch.Tensor): Query features in shape (batch_size, N, 256)
                - pred_boxes: (torch.Tensor): Normalized prediction boxes in shape (batch_size, N, 4),
                    in cxcywh format
        """
        input_images = nested_tensor_from_tensor_list(input_images)
        input_images = input_images.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_images, prompt_type)
        return outputs

    def construct_input(self, image: List[Union[str, Image.Image]]):
        """Construct input for the model

        Args:
            image (image: Union[List[Union[str, Image.Image]], torch.Tensor]): A list of image path or
                PIL.Image.Image object. If the length of the list is more than 1, the model w`ill
                perform batch inference.

        Returns:
            Tuple[torch.Tensor, List[List[int]]]: A tuple containing the
                input images, and the sizes of the input images.
        """
        input_images = []
        image_sizes = []
        for _, img in enumerate(image):
            if isinstance(img, str):
                img = Image.open(img)
            elif isinstance(img, Image.Image):
                img = img
            else:
                raise ValueError(
                    "image must be either a string or a PIL.Image.Image object"
                )
            W, H = img.size
            image_sizes.append([H, W])
            # add image in tensor format
            input_images.append(self.transform_image(img))
        return input_images, image_sizes

    def transform_image(self, image_pil: Image) -> Image:
        """apply a set of transformations to a cv2 load image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            Tuple[PIL.Image, torch.Tensor]: A tuple containing the original PIL Image and the
                transformed image as a PyTorch tensor.
        """
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        transformed_image, _ = transform(image_pil, None)  # 3, h, w
        return transformed_image

    def postprocess(
        self,
        outputs: Dict[str, torch.Tensor],
        image_pil_sizes: List[List[int]] = None,
    ):
        boxes = outputs["pred_boxes"].cpu()
        scores = (
            outputs["pred_logits"].sigmoid().cpu()
            if "pred_logits" in outputs
            else None
        )
        normalized_xyxy_boxes = []
        original_xyxy_boxes = []
        for batch_idx, (H, W) in enumerate(image_pil_sizes):
            batch_boxes = boxes[batch_idx]  # (num_queries, 4)
            # from (cx, cy, w, h) to (x1, y1, x2, y2)
            batch_boxes[:, 0] = batch_boxes[:, 0] - batch_boxes[:, 2] / 2
            batch_boxes[:, 1] = batch_boxes[:, 1] - batch_boxes[:, 3] / 2
            batch_boxes[:, 2] = batch_boxes[:, 0] + batch_boxes[:, 2]
            batch_boxes[:, 3] = batch_boxes[:, 1] + batch_boxes[:, 3]
            normalized_xyxy_boxes.append(copy.deepcopy(batch_boxes))
            #  scale boxes
            original_boxes = (
                batch_boxes.clone()
            )  # Copy the normalized boxes to scale to original sizes
            original_boxes[:, 0] = original_boxes[:, 0] * W
            original_boxes[:, 1] = original_boxes[:, 1] * H
            original_boxes[:, 2] = original_boxes[:, 2] * W
            original_boxes[:, 3] = original_boxes[:, 3] * H
            original_xyxy_boxes.append(original_boxes)

        original_xyxy_boxes = torch.stack(original_xyxy_boxes)
        original_xyxy_boxes = original_xyxy_boxes.numpy()

        # sort everything by score from highest to lowest
        sorted_original_boxes = []
        sorted_scores = []
        for i in range(len(normalized_xyxy_boxes)):
            scores_i = scores[i] if scores is not None else None
            # sort in descending order
            sorted_indices = scores_i.squeeze(-1).argsort(descending=True)
            sorted_original_boxes.append(
                original_xyxy_boxes[i][sorted_indices]
            )
            sorted_scores.append(scores_i[sorted_indices])

        original_xyxy_boxes = np.stack(sorted_original_boxes)
        scores = torch.stack(sorted_scores)

        return dict(
            original_xyxy_boxes=original_xyxy_boxes,
            scores=scores,
        )

    def filter(self, result: Dict, min_score: float, nms_value: float = 0.8):
        """Filter the UPN detection result. Only keep boxes with score above min_score
        and apply Non-Maximum Suppression (NMS) to filter overlapping boxes.

        Args:
            result (Dict): A dictionary containing detection results with 'original_xyxy_boxes' and 'scores'.
            min_score (float): Minimum score threshold for keeping a box.
            nms_value (float): NMS threshold for filtering boxes.

        Returns:
            Dict: Filtered result containing 'original_xyxy_boxes' and 'scores' with the filtered boxes.
        """
        filtered_result = {"original_xyxy_boxes": [], "scores": []}

        for boxes, scores in zip(
            np.array(result["original_xyxy_boxes"]), result["scores"].numpy()
        ):
            # Filter out boxes with score below min_score
            keep = scores >= min_score
            boxes = boxes[keep[:, 0]]
            scores = scores[keep[:, 0]][:, 0]

            if len(boxes) == 0:
                return filtered_result

            # Convert to torch tensors
            boxes = torch.tensor(boxes, dtype=torch.float32)
            scores = torch.tensor(scores, dtype=torch.float32)

            # Apply Non-Maximum Suppression (NMS)
            if nms_value > 0:
                keep_indices = nms(boxes, scores, nms_value)
            else:
                keep_indices = torch.arange(len(boxes))

            # Keep only the boxes that passed NMS
            filtered_boxes = boxes[keep_indices].numpy().astype(np.int32)
            filtered_scores = scores[keep_indices].numpy()

            # Sort the boxes by score in descending order
            sorted_indices = np.argsort(filtered_scores)[::-1]
            filtered_boxes = filtered_boxes[sorted_indices]
            filtered_scores = filtered_scores[sorted_indices]

            # Round the scores to two decimal places
            filtered_scores = [round(score, 2) for score in filtered_scores]

            # Store the filtered boxes and scores in the result dictionary
            filtered_result["original_xyxy_boxes"].append(
                filtered_boxes.tolist()
            )
            filtered_result["scores"].append(filtered_scores)

        return filtered_result
