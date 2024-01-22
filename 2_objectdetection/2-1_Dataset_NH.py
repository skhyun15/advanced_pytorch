import glob
import os.path as osp
import random
from typing import Any
import numpy as np
import json
from PIL import Image

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms


def make_datapath_list(rootpath):
    imgpath_template = osp.join(rootpath, "JPEGImages", "%s.jpg")
    annopath_template = osp.join(rootpath, "Annotations", "%s.xml")

    train_id_names = osp.join(rootpath, "ImageSets/Main/train.txt")

    val_id_names = osp.join(rootpath, "ImageSets/Main/val.txt")

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()
        img_path = imgpath_template % file_id
        anno_path = annopath_template % file_id
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()
        img_path = imgpath_template % file_id
        anno_path = annopath_template % file_id
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


rootpath = "./2_objectdetection/data/VOCdevkit/VOC2012/"

train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
    rootpath
)
print(train_img_list[0])
