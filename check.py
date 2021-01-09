import pandas as pd
import numpy as np
import cv2
import os, glob
import albumentations as albu
from albumentations.pytorch import ToTensorV2

import torch
from torch.nn import functional as F

from src.models import Timm_model
from src.losses import MyLabelSmoothingLoss

pd.set_option('display.max_rows', None)

# weight = 0.05
#
# target = torch.randint(0, 5, (8, 1))
# pred = torch.randn((8, 5))
# print(target)
# target = F.one_hot(target, num_classes=5)
#
# print(target)
#
# import torchvision
#
#
# total_trainset = torchvision.datasets.CIFAR10(root="./dataset/CIFAR-10", train=True,  download=True)
# train_labels = total_trainset.targets
#
# print(train_labels)


a = pd.read_csv('./input/probability.csv')

a = a.sort_values(by='pred')

print(a.head(1000))