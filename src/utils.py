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


def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x)*(Y_m - top_y) >=0)] = 1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


class CassavaDataset(Dataset):
    def __init__(self, data_dir, transform=None, phase='train', df=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.phase = phase
        self.img_path = glob.glob(os.path.join(self.data_dir, 'train_images', '*.jpg'))

    def __len__(self):
        if self.df is None:
            return len(self.img_path)
        else:
            return len(self.df)

    def __getitem__(self, idx):
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

        label = self.df[self.df['image_id'] == img_id]['label'].values
        label = torch.tensor(label, dtype=torch.long)

        return img, label
