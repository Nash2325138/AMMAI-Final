import os
import sys
import random
from glob import glob
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
        normalization=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    ):
        self.data_root = data_root
        self.test_transform = transforms.Compose([
            transforms.Resize([112, 112]),
            transforms.ToTensor(),
            transforms.Normalize(*normalization)
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


class AsiaCelebDataLoader(BaseDataLoader):
    def __init__(
        self, batch_size,
        shuffle, validation_split,
        num_workers, dataset_args={},
        name='train'
    ):
        self.name = name
        self.dataset = AsiaCelebDataset(**dataset_args)
        self.num_classes = self.dataset.num_classes
        super().__init__(
            self.dataset, batch_size, shuffle,
            validation_split, num_workers)


class AsiaCelebDataset(Dataset):
    """
    A wrapper yeilding dictionary for ImageFolder dataste of Asis-celab.
    """
    def __init__(
        self, data_root, normalization=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        uniform_on_person=False, min_n_faces=10
    ):
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*normalization)
        ])
        self.data_root = data_root

        # statics about image folders
        folders = sorted(glob(os.path.join(data_root, '*')))
        n_faces = [len(glob(os.path.join(folder, '*'))) for folder in folders]

        # filter out folders whose #images < min_n_faces
        valid_folder_idxs = [i for i, n in enumerate(n_faces) if n >= min_n_faces]

        self.folders = [folders[i] for i in valid_folder_idxs]
        self.n_faces = [n_faces[i] for i in valid_folder_idxs]
        self.num_classes = len(valid_folder_idxs)

        self.uniform_on_person = uniform_on_person
        if uniform_on_person:
            self.length = self.num_classes
        else:
            self.length = sum(self.n_faces)

    def __getitem__(self, index):
        if self.uniform_on_person:
            face_idx = random.randint(0, self.n_faces[index] - 1)
            face_path = os.path.join(self.folders[index], f'{face_idx}.jpg')
            face_image = Image.open(face_path)
            face_tensor = self.train_transform(face_image)
            return {
                'face_tensor': face_tensor,
                'faceID': index
            }
        else:
            raise NotImplementedError()

    def __len__(self):
        return self.length
