# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args, datasetinfo):
    if datasetinfo["dataset_mode"] == 'coco':
        return build_coco(image_set, args, datasetinfo)
    if datasetinfo["dataset_mode"] == 'odvg':
        from .odvg import build_odvg
        return build_odvg(image_set, args, datasetinfo)
    raise ValueError(f'dataset {args.dataset_file} not supported')
