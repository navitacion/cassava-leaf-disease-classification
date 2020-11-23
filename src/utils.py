import os
import cv2
import glob
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class ImageTransform:
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = {
            'train': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0),

            'val': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0),

            'test': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0)
        }

    def __call__(self, img, phase='train'):
        augmented = self.transform[phase](image=img)
        augmented = augmented['image']

        return augmented



class CassavaDataset(Dataset):
    def __init__(self, data_dir, transform=None, phase='train', df=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.phase = phase
        if self.phase == 'test':
            img_dir = 'test_images'
        else:
            img_dir = 'train_images'
        self.img_path = glob.glob(os.path.join(self.data_dir, img_dir, '*.jpg'))

    def __len__(self):
        if self.df is None:
            return len(self.img_path)
        else:
            return len(self.df)

    def __getitem__(self, idx):

        if self.phase == 'test':
            target_img_path = self.img_path[idx]
        else:
            row = self.df.iloc[idx]
            target_img_id = row['image_id']
            target_img_path = os.path.join(self.data_dir, 'train_images', f'{target_img_id}')

        img = cv2.imread(target_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_id = os.path.basename(target_img_path)

        if self.transform is not None:
            img = self.transform(img, self.phase)
        else:
            img = torch.from_numpy(img.transpose((2, 0, 1)))
            img = img / 255.

        if self.phase == 'test':
            return img, img_id

        else:
            label = self.df[self.df['image_id'] == img_id]['label'].values
            label = torch.tensor(label, dtype=torch.long)

            return img, label
