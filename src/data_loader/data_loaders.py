import os
import json
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # noqa

from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np
import torch

from base.base_data_loader import BaseDataLoader
from utils.logging_config import logger


class AsiaLegisDataLoader(BaseDataLoader):
    def __init__(
        self, batch_size,
        shuffle, validation_split,
        num_workers, dataset_args={},
        name='train'
    ):
        self.name = name
        self.dataset = AsiaLegisDataSet(**dataset_args)
        super().__init__(
            self.dataset, batch_size, shuffle,
            validation_split, num_workers)

    @property
    def num_classes(self):
        raise NotImplementedError()


class AsiaLegisDataSet(Dataset):
    def __init__(
        self, data_root,
        augmentation=True
    ):
        self.data_root = data_root
        self.augmentation = augmentation
        self.test_trasnform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])  # Following https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/config.py
        self.data_table = self.read_pairs()

    def read_pairs():
        with open(os.path.join(self.data_root, "pairs.txt"), 'r') as f:
            lines = f.readlines()
        data_table = []
        for line in lines:
            splits = line.split()
            if len(splits) == 3:
                entry = (True, splits[0], splits[1], splits[0], splits[2])
            elif len(splits) == 4:
                entry = (False, splits[0], splits[1], splits[2], splits[3])
            else:
                raise RuntimeError("Each line must have 3 or 4 entities.")
            data_table.append(entry)
        return data_table

    def __getitem__(self, index):

    def __len__(self):
        return len(self.data_table)
