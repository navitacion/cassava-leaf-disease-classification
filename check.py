import pandas as pd
import numpy as np
import cv2
import os, glob
import albumentations as albu
from albumentations.pytorch import ToTensorV2

import torch

from src.models import Timm_model

# Image Augmentations
class ImageTransform:
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = albu.Compose([
            albu.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            albu.Normalize(mean, std),
            ToTensorV2(),
        ], p=1.0)

    def __call__(self, img):
        augmented = self.transform(image=img)
        augmented = augmented['image']

        return augmented

train = pd.read_csv('./input/relabel.csv')

print(train.shape)
train = train[train['rename'] != 5]
print(train.shape)

print(train['label'].value_counts())

train.loc[(train['label'] != 3)&(train['rename'] == 3), 'label'] = 3

print(train['label'].value_counts())

train = train[['image_id', 'label']]

train.to_csv('./input/train_change_only3.csv', index=False)

# train = pd.read_csv('./input/train.csv')
#
# print(train.shape)
#
# print(train.head())
#
# train = train.drop(['label', 'rename'], axis=1)
# train = train.rename(columns={'relabel': 'label'})
#
# train.to_csv('./input/train_2.csv', index=False)


# model_weights = sorted(glob.glob('./weights/*'))
#
# all_res = pd.DataFrame()
# all_res['image_id'] = train['image_id']
# all_res['label'] = train['label']
#
# for fold, w in enumerate(model_weights):
#     net = Timm_model('efficientnet_b0', pretrained=False)
#     net.load_state_dict(torch.load(w))
#     net.eval()
#     c = 0
#
#     for i in range(len(train)):
#
#         row = train.iloc[i]
#         image_id = row['image_id']
#         label = row['label']
#
#         img_path = f'./input/train_images/{image_id}'
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         transform = ImageTransform(img_size=512)
#         img = transform(img)
#
#         out = torch.softmax(net(img.unsqueeze(0)), 1)
#         out = out.detach().cpu().numpy()
# sudo sersu
#         if c == 0:
#             res = out
#         else:
#             res = np.concatenate([res, out], 0)
#
#         c += 1
#
#     df = pd.DataFrame(res, columns=[f'fold_{fold}_label_{c}' for c in range(5)])
#     df['image_id'] = train['image_id'].values
#     df['label'] = train['label'].values
#
#     all_res = all_res.merge(df, on=['image_id', 'label'])
#
#
# all_res.to_csv('predicted.csv', index=False)