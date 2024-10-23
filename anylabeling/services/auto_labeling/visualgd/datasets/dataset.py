from __future__ import print_function

import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from PIL import Image
from .tsv_io import TSVFile
import numpy as np
import base64
import io


class TSVDataset(Dataset):
    """ TSV dataset for ImageNet 1K training
    """    
    def __init__(self, tsv_file, transform=None, target_transform=None):
        self.tsv = TSVFile(tsv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        row = self.tsv.seek(index)
        image_data = base64.b64decode(row[-1])
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        target = int(row[1])

        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.tsv.num_rows()
