import pandas as pd
import cv2
import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as albu
from albumentations.pytorch import ToTensorV2

from src.models import Timm_model
from src.utils import CassavaDataset

# Config
batch_size = 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('./input/merged.csv')
data_dir = './input'

class ImageTransform:
    def __init__(self, img_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = {'train': albu.Compose([albu.Resize(img_size, img_size),
                                       albu.Normalize(mean, std),
                                       ToTensorV2()
                                       ], p=1.0)}

    def __call__(self, img, phase='train'):
        augmented = self.transform[phase](image=img)
        augmented = augmented['image']

        return augmented

transform = ImageTransform()
weights = glob.glob('./weights/efnb4_01*')
models = []

for w in weights:
    model = Timm_model(model_name='tf_efficientnet_b4_ns', pretrained=False)
    model.load_state_dict(torch.load(w))
    model = model.to(device)
    model.eval()
    models.append(model)

preds = []
img_ids = []

# Dataset
train_dataset = CassavaDataset(data_dir, transform, phase='train', df=df)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

for img, label, img_id in tqdm(train_dataloader):
    img = img.to(device)
    label = label.to(device)
    label = F.one_hot(label, num_classes=5)
    pred = torch.zeros((label.size(0), 5))
    pred = pred.to(device)

    for m in models:
        pred += F.softmax(m(img), dim=1) / len(models)

    # labelのアダマール積を取る
    out = pred * label.squeeze()
    # 予測ラベルにおける確率を計算する
    out = torch.sum(out, dim=1).detach().cpu().numpy()

    preds.extend(out)
    img_ids.extend(list(img_id))


res = pd.DataFrame({
    'image_id': img_ids,
    'pred': preds
})

res.to_csv('./input/probability.csv', index=False)