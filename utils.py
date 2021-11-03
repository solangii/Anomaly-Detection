"""This module has some useful functions"""

import os
import random

import numpy as np
import torch


def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    elif not os.path.exists(paths):
        os.makedirs(paths)


def compare_images_colab(real_img, generated_img, data, threshold=0.4):
    diff_img = np.abs(generated_img - real_img)
    threshold = threshold * 256.
    diff_img[diff_img <= threshold] = 0

    anomaly_img = np.zeros_like(real_img)
    # anomaly_img[:, :, :] = real_img
    anomaly_img[np.where(diff_img > 0)[0], np.where(diff_img > 0)[1]] = [0, 0, 200]

    return anomaly_img


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_save_path(config):
    save_path = os.path.join(config.save_root, config.dataset)

    if config.mode == 'train':
        # train mode needs path for save parameter
        save_path = os.path.join(save_path, 'param')
    else:
        # test mode needs path for save image
        save_path = os.path.join(save_path, 'img')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('make directory ...', str(save_path))

    return save_path


def exp_name(epoch, lr, bs, c):
    name = ""
    name += "epo-" + str(epoch)
    name += "_lr-" + str(lr)
    name += "_bs-" + str(bs)
    name += "_const-" + str(c)
    return name
