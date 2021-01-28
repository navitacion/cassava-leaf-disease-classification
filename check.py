import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, glob
import albumentations as albu
from albumentations.pytorch import ToTensorV2

import torch
from torch.nn import functional as F

from src.models import Timm_model
from src.losses import MyLabelSmoothingLoss
import torchvision.datasets as datasets

# pd.set_option('display.max_rows', None)
#
#
# csv_list = glob.glob(os.path.join('./input/oof', '*.csv'))
#
# print(csv_list)
#
# oof = pd.DataFrame()
#
# for path in csv_list:
#     tmp = pd.read_csv(path)
#     oof = pd.concat([oof, tmp], axis=0)
#
#
# label_cols = sorted([c for c in oof.columns if c.startswith('pred_label')])
# print(label_cols)
# oof['pred_label'] = np.argmax(oof[label_cols].values, axis=1)
# oof['max_probability'] = np.max(oof[label_cols].values, axis=1)
#
#
# from sklearn.metrics import confusion_matrix
#
# matrix = confusion_matrix(y_true=oof['label'], y_pred=oof['pred_label'])
#
# import seaborn as sns
# sns.heatmap(matrix, annot=True, fmt="d", cbar=False)
# plt.ylabel('True')
# plt.xlabel('Predicted')
#
# plt.savefig('t.png')
# plt.show()
#
#
# tar = oof.query("pred_label != label and max_probability > 0.8")
#
# print(f'Drop Noise: {len(tar)} / {len(oof)}')
#
# print(tar['label'].value_counts() / len(tar))


a = [-0.531, 2.56, 1.83, 0.06]
a = torch.tensor(a)

out = F.softmax(a)

print(out)
