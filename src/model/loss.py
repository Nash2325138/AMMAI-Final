import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, output_key='logits', target_key='faceID'):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, data, output):
        logits = output[self.output_key]
        target = data[self.target_key]
        return self.loss_fn(logits, target)


class UnifromFaceLoss(nn.Module):
    def __init__(self, num_classes, emb_dim=512):
        self.num_classes
        self.centers = torch.zeros([num_classes, emb_dim])

    def _delta_center(data, output):
        pass

    def forward(self, data, output):
        """
        1. Calculate delta (i.e. update values) of centers.
        2. Based on updated centers, calculate L_u
        """
        pass
