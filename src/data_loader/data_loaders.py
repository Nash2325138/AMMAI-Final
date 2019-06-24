import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # noqa

from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms

from base.base_data_loader import BaseDataLoader
# from utils.logging_config import logger


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
    ):
        self.data_root = data_root
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])  # Following https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/config.py
        self.data_table = self.read_pairs()

    def read_pairs(self):
        """
        Read pairs.txt to build a data table where each entry has a following structure:
        (is_the_same: bool, person1: str, faceID1: str, person2: str, faceID2: str)
        """
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

    def get_image(self, person, faceID):
        return Image.open(os.path.join(self.data_root, 'C', f'{person}_{faceID}.jpg'))

    def __getitem__(self, index):
        """
        Read data in data_table[index] and return a dictionary:
            {
                'f1': tensor of the normalized face 1
                'f2': tensor of the normalized face 2
                'is_same': whether f1 and f2 have the same identity (bool)
            }
        """
        entry = self.data_table[index]
        f1_image = self.get_image(entry[1], entry[2])
        f2_image = self.get_image(entry[3], entry[4])
        is_same = entry[0]
        return {
            'f1': self.test_transform(f1_image),
            'f2': self.test_transform(f2_image),
            'is_same': is_same
        }

    def __len__(self):
        return len(self.data_table)
