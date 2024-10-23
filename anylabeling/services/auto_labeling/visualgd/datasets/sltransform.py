# modified from https://github.com/anhtuan85/Data-Augmentation-for-Object-Detection/blob/master/augmentation.ipynb

import PIL #version 1.2.0
from PIL import Image #version 6.1.0
import torch
import os
import torchvision.transforms.functional as F
import numpy as np
import random

from .random_crop import random_crop
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

class AdjustContrast:
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, img, target):
        """
        img (PIL Image or Tensor): Image to be adjusted.
        """
        _contrast_factor = ((random.random() + 1.0) / 2.0) * self.contrast_factor
        img = F.adjust_contrast(img, _contrast_factor)
        return img, target

class AdjustBrightness:
    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, img, target):
        """
        img (PIL Image or Tensor): Image to be adjusted.
        """
        _brightness_factor = ((random.random() + 1.0) / 2.0) * self.brightness_factor
        img = F.adjust_brightness(img, _brightness_factor)
        return img, target

def lighting_noise(image):
    '''
        color channel swap in image
        image: A PIL image
    '''
    new_image = image
    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), 
             (1, 2, 0), (2, 0, 1), (2, 1, 0))
    swap = perms[random.randint(0, len(perms)- 1)]
    new_image = F.to_tensor(new_image)
    new_image = new_image[swap, :, :]
    new_image = F.to_pil_image(new_image)
    return new_image

class LightingNoise:
    def __init__(self) -> None:
        pass

    def __call__(self, img, target):
        return lighting_noise(img), target


def rotate(image, boxes, angle):
    '''
        Rotate image and bounding box
        image: A Pil image (w, h)
        boxes: A tensors of dimensions (#objects, 4)
        
        Out: rotated image (w, h), rotated boxes
    '''
    new_image = image.copy()
    new_boxes = boxes.clone()
    
    #Rotate image, expand = True
    w = image.width
    h = image.height
    cx = w/2
    cy = h/2
    new_image = new_image.rotate(angle, expand=True)
    angle = np.radians(angle)
    alpha = np.cos(angle)
    beta = np.sin(angle)
    #Get affine matrix
    AffineMatrix = torch.tensor([[alpha, beta, (1-alpha)*cx - beta*cy],
                                 [-beta, alpha, beta*cx + (1-alpha)*cy]])
    
    #Rotation boxes
    box_width = (boxes[:,2] - boxes[:,0]).reshape(-1,1)
    box_height = (boxes[:,3] - boxes[:,1]).reshape(-1,1)
    
    #Get corners for boxes
    x1 = boxes[:,0].reshape(-1,1)
    y1 = boxes[:,1].reshape(-1,1)
    
    x2 = x1 + box_width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + box_height
    
    x4 = boxes[:,2].reshape(-1,1)
    y4 = boxes[:,3].reshape(-1,1)
    
    corners = torch.stack((x1,y1,x2,y2,x3,y3,x4,y4), dim= 1)
    # corners.reshape(-1, 8)    #Tensors of dimensions (#objects, 8)
    corners = corners.reshape(-1,2) #Tensors of dimension (4* #objects, 2)
    corners = torch.cat((corners, torch.ones(corners.shape[0], 1)), dim= 1) #(Tensors of dimension (4* #objects, 3))
    
    cos = np.abs(AffineMatrix[0, 0])
    sin = np.abs(AffineMatrix[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    AffineMatrix[0, 2] += (nW / 2) - cx
    AffineMatrix[1, 2] += (nH / 2) - cy
    

    #Apply affine transform
    rotate_corners = torch.mm(AffineMatrix, corners.t().to(torch.float64)).t()
    rotate_corners = rotate_corners.reshape(-1,8)
    
    x_corners = rotate_corners[:,[0,2,4,6]]
    y_corners = rotate_corners[:,[1,3,5,7]]
    
    #Get (x_min, y_min, x_max, y_max)
    x_min, _ = torch.min(x_corners, dim= 1)
    x_min = x_min.reshape(-1, 1)
    y_min, _ = torch.min(y_corners, dim= 1)
    y_min = y_min.reshape(-1, 1)
    x_max, _ = torch.max(x_corners, dim= 1)
    x_max = x_max.reshape(-1, 1)
    y_max, _ = torch.max(y_corners, dim= 1)
    y_max = y_max.reshape(-1, 1)
    
    new_boxes = torch.cat((x_min, y_min, x_max, y_max), dim= 1)
    
    scale_x = new_image.width / w
    scale_y = new_image.height / h
    
    #Resize new image to (w, h)

    new_image = new_image.resize((w, h))
    
    #Resize boxes
    new_boxes /= torch.Tensor([scale_x, scale_y, scale_x, scale_y])
    new_boxes[:, 0] = torch.clamp(new_boxes[:, 0], 0, w)
    new_boxes[:, 1] = torch.clamp(new_boxes[:, 1], 0, h)
    new_boxes[:, 2] = torch.clamp(new_boxes[:, 2], 0, w)
    new_boxes[:, 3] = torch.clamp(new_boxes[:, 3], 0, h)
    return new_image, new_boxes

# def convert_xywh_to_xyxy(boxes: torch.Tensor):
#     _boxes = boxes.clone()
#     box_xy = _boxes[:, :2]
#     box_wh = _boxes[:, 2:]
#     box_x1y1 = box_xy - box_wh/2 
#     box_x2y2 = box_xy + box_wh/2
#     box_xyxy = torch.cat((box_x1y1, box_x2y2), dim=-1)
#     return box_xyxy

class Rotate:
    def __init__(self, angle=10) -> None:
        self.angle = angle

    def __call__(self, img, target):
        w,h = img.size
        whwh = torch.Tensor([w, h, w, h])
        boxes_xyxy = box_cxcywh_to_xyxy(target['boxes']) * whwh
        img, boxes_new = rotate(img, boxes_xyxy, self.angle)
        target['boxes'] = box_xyxy_to_cxcywh(boxes_new).to(boxes_xyxy.dtype) / (whwh + 1e-3)
        return img, target


class RandomCrop:
    def __init__(self) -> None:
        pass

    def __call__(self, img, target):
        w,h = img.size
        try:
            boxes_xyxy = target['boxes']
            labels = target['labels']
            img, new_boxes, new_labels, _ = random_crop(img, boxes_xyxy, labels)
            target['boxes'] = new_boxes
            target['labels'] = new_labels
        except Exception as e:
            pass
        return img, target


class RandomCropDebug:
    def __init__(self) -> None:
        pass

    def __call__(self, img, target):
        boxes_xyxy = target['boxes'].clone()
        labels = target['labels'].clone()
        img, new_boxes, new_labels, _ = random_crop(img, boxes_xyxy, labels)
        target['boxes'] = new_boxes
        target['labels'] = new_labels


        return img, target
        
class RandomSelectMulti(object):
    """
    Randomly selects between transforms1 and transforms2,
    """
    def __init__(self, transformslist, p=-1):
        self.transformslist = transformslist
        self.p = p
        assert p == -1

    def __call__(self, img, target):
        if self.p == -1:
            return random.choice(self.transformslist)(img, target)


class Albumentations:
    def __init__(self):
        import albumentations as A
        self.transform = A.Compose([
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.005),
            A.RandomGamma(p=0.005),
            A.ImageCompression(quality_lower=75, p=0.005)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, img, target, p=1.0):
        """
        Input:
            target['boxes']: xyxy, unnormalized data.
        
        """
        boxes_raw = target['boxes']
        labels_raw = target['labels']
        img_np = np.array(img)
        if self.transform and random.random() < p:
            new_res = self.transform(image=img_np, bboxes=boxes_raw, class_labels=labels_raw)  # transformed
            boxes_new = torch.Tensor(new_res['bboxes']).to(boxes_raw.dtype).reshape_as(boxes_raw)
            img_np = new_res['image']
            labels_new = torch.Tensor(new_res['class_labels']).to(labels_raw.dtype)
        img_new = Image.fromarray(img_np)
        target['boxes'] = boxes_new
        target['labels'] = labels_new
        
        return img_new, target