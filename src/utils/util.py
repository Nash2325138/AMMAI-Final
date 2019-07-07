import os
import io
from glob import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as trans
from PIL import Image
from .logging_config import logger


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_everything_under(root_dir, pattern='*', only_dirs=False, only_files=False):
    assert not(only_dirs and only_files), 'You will get nothnig '\
        'when "only_dirs" and "only_files" are both set to True'
    everything = sorted(glob(os.path.join(root_dir, pattern)))
    if only_dirs:
        everything = list(filter(lambda f: os.path.isdir(f), everything))
    if only_files:
        everything = list(filter(lambda f: os.path.isfile(f), everything))
    return everything


def one_hot_embedding(labels, num_classes):
    # From https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/26
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def replace_module_prefix(state_dict, prefix='module.', replace=''):
    new_state = {}
    for key in state_dict.keys():
        if key.startswith(prefix):
            new_key = replace + key[len(prefix):]
        else:
            new_key = key
        new_state[new_key] = state_dict[key]
    return new_state


def extract_missing_and_unexpected_keys(source_keys, target_keys):
    unexpected = [key for key in source_keys if key not in target_keys]
    missing = [key for key in target_keys if key not in source_keys]
    return missing, unexpected


def softmax(data):
    data = torch.tensor(np.array(data))
    data = torch.softmax(data, dim=1).numpy()
    assert np.isclose(data[0].sum(), 1)
    assert np.isclose(data[-1].sum(), 1)
    return data


def gen_roc_plot(fpr, tpr, return_tensor=False, save_to=None):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    if save_to:
        logger.info(f'Saving ROC curve to {save_to}')
        plt.savefig(save_to)
    buf.seek(0)
    plt.close()
    if return_tensor:
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return roc_curve_tensor
    else:
        return buf


def gen_acc_thres_plot(thres, accs, return_tensor=False, save_to=None):
    plt.figure()
    plt.xlabel("threshold", fontsize=14)
    plt.ylabel("accuracy", fontsize=14)
    plt.plot(thres, accs, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    if save_to:
        logger.info(f'Saving accuracy curve to {save_to}')
        plt.savefig(save_to)
    buf.seek(0)
    plt.close()
    if return_tensor:
        curve = Image.open(buf)
        curve_tensor = trans.ToTensor()(curve)
        return curve_tensor
    else:
        return buf
