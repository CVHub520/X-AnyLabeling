import PIL #version 1.2.0
import torch
import os
import torchvision.transforms.functional as F
import numpy as np
import random


def intersect(boxes1, boxes2):
    '''
        Find intersection of every box combination between two sets of box
        boxes1: bounding boxes 1, a tensor of dimensions (n1, 4)
        boxes2: bounding boxes 2, a tensor of dimensions (n2, 4)
        
        Out: Intersection each of boxes1 with respect to each of boxes2, 
             a tensor of dimensions (n1, n2)
    '''
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_xy =  torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
                        boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))
    
    min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))
    inter = torch.clamp(max_xy - min_xy , min=0)  # (n1, n2, 2)
    return inter[:, :, 0] * inter[:, :, 1]  #(n1, n2)
def find_IoU(boxes1, boxes2):
    '''
        Find IoU between every boxes set of boxes 
        boxes1: a tensor of dimensions (n1, 4) (left, top, right , bottom)
        boxes2: a tensor of dimensions (n2, 4)
        
        Out: IoU each of boxes1 with respect to each of boxes2, a tensor of 
             dimensions (n1, n2)
        
        Formula: 
        (box1 ∩ box2) / (box1 u box2) = (box1 ∩ box2) / (area(box1) + area(box2) - (box1 ∩ box2 ))
    '''
    inter = intersect(boxes1, boxes2)
    area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    area_boxes1 = area_boxes1.unsqueeze(1).expand_as(inter) #(n1, n2)
    area_boxes2 = area_boxes2.unsqueeze(0).expand_as(inter)  #(n1, n2)
    union = (area_boxes1 + area_boxes2 - inter)
    return inter / union


def random_crop(image, boxes, labels, difficulties=None):
    '''
        image: A PIL image
        boxes: Bounding boxes, a tensor of dimensions (#objects, 4)
        labels: labels of object, a tensor of dimensions (#objects)
        difficulties: difficulties of detect object, a tensor of dimensions (#objects)
        
        Out: cropped image , new boxes, new labels, new difficulties
    '''
    if type(image) == PIL.Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    
    while True:
        mode = random.choice([0.1, 0.3, 0.5, 0.9, None])
        
        if mode is None:
            return F.to_pil_image(image), boxes, labels, difficulties
        
        new_image = image
        new_boxes = boxes
        new_difficulties = difficulties
        new_labels = labels
        for _ in range(50):
            # Crop dimensions: [0.3, 1] of original dimensions
            new_h = random.uniform(0.3*original_h, original_h)
            new_w = random.uniform(0.3*original_w, original_w)
            
            # Aspect ratio constraint b/t .5 & 2
            if new_h/new_w < 0.5 or new_h/new_w > 2:
                continue
            
            #Crop coordinate
            left = random.uniform(0, original_w - new_w)
            right = left + new_w
            top = random.uniform(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([int(left), int(top), int(right), int(bottom)])
            
            # Calculate IoU  between the crop and the bounding boxes
            overlap = find_IoU(crop.unsqueeze(0), boxes) #(1, #objects)
            overlap = overlap.squeeze(0)

            # If not a single bounding box has a IoU of greater than the minimum, try again
            if overlap.shape[0] == 0:
                continue
            if overlap.max().item() < mode:
                continue
            
            #Crop
            new_image = image[:, int(top):int(bottom), int(left):int(right)] #(3, new_h, new_w)
            
            #Center of bounding boxes
            center_bb = (boxes[:, :2] + boxes[:, 2:])/2.0
            
            #Find bounding box has been had center in crop
            center_in_crop = (center_bb[:, 0] >left) * (center_bb[:, 0] < right
                             ) *(center_bb[:, 1] > top) * (center_bb[:, 1] < bottom)    #( #objects)
            
            if not center_in_crop.any():
                continue
            
            #take matching bounding box
            new_boxes = boxes[center_in_crop, :]
            
            #take matching labels
            new_labels = labels[center_in_crop]
            
            #take matching difficulities
            if difficulties is not None:
                new_difficulties = difficulties[center_in_crop]
            else:
                new_difficulties = None
            
            #Use the box left and top corner or the crop's
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])
            
            #adjust to crop
            new_boxes[:, :2] -= crop[:2]
            
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:],crop[2:])
            
            #adjust to crop
            new_boxes[:, 2:] -= crop[:2]
            
            return F.to_pil_image(new_image), new_boxes, new_labels, new_difficulties