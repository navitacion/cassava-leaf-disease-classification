import os
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import metrics

from .utils import CassavaDataset
from .utils import StratifiedSampler
from .cutmix import cutmix, CutMixCriterion, resizemix
from .mixup import mixup, MixupCriterion
from .snapmix import snapmix, SnapMixLoss


# 目視で確認
drop_images = [
    '9224019.jpg',
    '102968016.jpg',
    '159654644.jpg',
    '199112616.jpg',
    '262902341.jpg',
    '313266547.jpg',
    '314640668.jpg',
    '357924077.jpg',
    '421035788.jpg',
    '479472063.jpg',
    '490603548.jpg',
    '520111872.jpg',
    '549854027.jpg',
    '554488826.jpg',
    '580111608.jpg',
    '600736721.jpg',
    '616718743.jpg',
    '695438825.jpg',
    '723564013.jpg',
    '744383303.jpg',
    '746746526.jpg',
    '826231979.jpg',
    '835290707.jpg',
    '847847826.jpg',
    '873637313.jpg',
    '888983519.jpg',
    '992748624.jpg',
    '1004389140.jpg',
    '1008244905.jpg',
    '1010470173.jpg',
    '1014492188.jpg',
    '1119403430.jpg',
    '1338159402.jpg',
    '1339403533.jpg',
    '1357797590.jpg',
    '1359893940.jpg',
    '1366430957.jpg',
    '1403621003.jpg',
    '1689510013.jpg',
    '1770746162.jpg',
    '1773381712.jpg',
    '1819546557.jpg',
    '1841279687.jpg',
    '1848686439.jpg',
    '1862072615.jpg',
    '1960041118.jpg',
    '2074713873.jpg',
    '2084868828.jpg',
    '2099754293.jpg',
    '2161797110.jpg',
    '2182500020.jpg',
    '2213446334.jpg',
    '2229847111.jpg',
    '2278166989.jpg',
    '2282957832.jpg',
    '2321669192.jpg',
    '2382642453.jpg',
    '2445684335.jpg',
    '2482667092.jpg',
    '2484530081.jpg',
    '2489013604.jpg',
    '2604713994.jpg',
    '2642216511.jpg',
    '2719114674.jpg',
    '2839068946.jpg',
    '3040241097.jpg',
    '3058561440.jpg',
    '3126296051.jpg',
    '3251960666.jpg',
    '3252232501.jpg',
    '3421208425.jpg',
    '3425850136.jpg',
    '3435954655.jpg',
    '3477169212.jpg',
    '3609350672.jpg',
    '3609986814.jpg',
    '3652033201.jpg',
    '3724956866.jpg',
    '3746679490.jpg',
    '3838556102.jpg',
    '3892366593.jpg',
    '3966432707.jpg',
    '4060987360.jpg',
    '4089218356.jpg',
    '4269208386.jpg',
]

class CassavaDataModule(pl.LightningDataModule):
    """
    DataModule for Cassava Competition
    """
    def __init__(self, data_dir, cfg, transform, cv,
                 use_merge=False, drop_noise=False, sample=False):
        """
        ------------------------------------
        Parameters
        data_dir: str
            Directory Path of Data
        cfg: DictConfig
            Config
        transform: albumentations.transform
            Data Augumentations
        cv: sklearn.model_selection
            Cross Validation
        """
        super(CassavaDataModule, self).__init__()
        self.data_dir = data_dir
        self.cfg = cfg
        self.transform = transform
        self.cv = cv
        self.use_merge = use_merge
        self.drop_noise = drop_noise
        self.sample = sample

    def prepare_data(self):
        # Prepare Data
        if self.use_merge:
            self.df = pd.read_csv(os.path.join(self.data_dir, 'merged.csv'))
        else:
            self.df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))

        if self.drop_noise:
            # 予測結果からかけ離れたものを除外
            # threshhold = 0.01
            # probability = pd.read_csv(os.path.join(self.data_dir, 'probability.csv'))
            # probability = probability[probability['pred'] > threshhold]
            # use_image_ids = probability['image_id'].values
            # self.df = self.df[self.df['image_id'].isin(use_image_ids)].reset_index(drop=True)

            # 手動で確認したものを除外
            self.df = self.df[~self.df['image_id'].isin(drop_images)].reset_index(drop=True)

        # 学習高速化のためにデータを1/3に分割
        if self.sample:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
            self.df['fold'] = -1
            for i, (trn_idx, val_idx) in enumerate(cv.split(self.df, self.df['label'])):
                self.df.loc[val_idx, 'fold'] = i
            self.df = self.df[self.df['fold'] == 0].reset_index(drop=True)
            self.df = self.df.drop(['fold'], axis=1)

    def setup(self, stage=None):
        # Validation
        self.df['fold'] = -1
        for i, (trn_idx, val_idx) in enumerate(self.cv.split(self.df, self.df['label'])):
            self.df.loc[val_idx, 'fold'] = i
        fold = self.cfg.train.fold
        train = self.df[self.df['fold'] != fold].reset_index(drop=True)
        val = self.df[self.df['fold'] == fold].reset_index(drop=True)

        # Dataset
        self.train_dataset = CassavaDataset(self.data_dir, self.transform, phase='train', df=train)
        self.val_dataset = CassavaDataset(self.data_dir, self.transform, phase='val', df=val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                              batch_size=self.cfg.train.batch_size,
                              pin_memory=False,
                              num_workers=self.cfg.train.num_workers,
                              # sampler=StratifiedSampler(self.train_dataset.df['label'].values),
                              shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.train.batch_size,
                          pin_memory=False,
                          num_workers=self.cfg.train.num_workers,
                          shuffle=False)


class CassavaLightningSystem(pl.LightningModule):
    def __init__(self, net, cfg, criterion, optimizer, scheduler=None, experiment=None):
        """
        ------------------------------------
        Parameters
        net: torch.nn.Module
            Model
        cfg: DictConfig
            Config
        optimizer: torch.optim
            Optimizer
        scheduler: torch.optim.lr_scheduler
            Learning Rate Scheduler
        experiment: comet_ml.experiment
            Logger(Comet_ML)
        """
        super(CassavaLightningSystem, self).__init__()
        self.net = net
        self.cfg = cfg
        self.experiment = experiment
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = 1e+9
        self.best_acc = 0
        self.epoch_num = 0
        self.acc_fn = metrics.Accuracy()

    def configure_optimizers(self):
        if self.scheduler is None:
            return [self.optimizer], []
        else:
            return [self.optimizer], [self.scheduler]

    def forward(self, x):
        output = self.net(x)
        return output

    def step(self, batch, phase='train', rand=None, packed=None):
        if rand is None:
            rand = np.random.rand()

        if packed is None:
            inp, label, img_id = batch
        else:
            inp, label = packed

        # Cutmixを徐々にへらす
        # th = self.current_epoch * 0.1
        # Cutmixはずっと一定に発生
        th = 1.0

        # Cutmix
        if rand > (th - self.cfg.train.cutmix_pct) and phase == 'train':
            if packed is None:
                inp, label = cutmix(inp, label, alpha=self.cfg.train.cutmix_alpha)
                packed = (inp, label)
            else:
                pass
            out = self.forward(inp)
            loss_fn = CutMixCriterion(criterion_base=self.criterion)
            loss = loss_fn(out, label)

        # ResizeMix
        elif rand > (th - self.cfg.train.resizemix_pct) and phase == 'train':
            inp, label = resizemix(inp, label, alpha=self.cfg.train.resizemix_alpha)
            out = self.forward(inp)
            loss_fn = CutMixCriterion(criterion_base=self.criterion)
            loss = loss_fn(out, label)

        # Mixup
        elif rand > (1.0 - self.cfg.train.mixup_pct) and phase == 'train':
            if packed is None:
                inp, label = mixup(inp, label, alpha=self.cfg.train.mixup_alpha)
                packed = (inp, label)
            else:
                _inp, label = packed
            out = self.forward(inp)
            loss_fn = MixupCriterion(criterion_base=self.criterion)
            loss = loss_fn(out, label)
        else:
            out = self.forward(inp)
            loss = self.criterion(out, label.squeeze())

        return loss, label, F.softmax(out, dim=1), img_id, rand, packed

    def training_step(self, batch, batch_idx):
        # SAM Optimizer - Second Time Manual Backward
        if self.cfg.train.use_sam:
            opt = self.optimizers()
            loss_1, _, _, _, rand, packed = self.step(batch, phase='train')
            self.manual_backward(loss_1, opt)
            opt.first_step(zero_grad=True)

            loss_2, _, _, _, _, _ = self.step(batch, phase='train', rand=rand, packed=packed)
            self.manual_backward(loss_2, opt)
            opt.second_step(zero_grad=True)

            loss = (loss_1 + loss_2) / 2

        # Default Optimizer  Once Auto Backward
        else:
            loss, _, _, _, _, _ = self.step(batch, phase='train')

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, label, logits, img_id, _, _ = self.step(batch, phase='val')

        return {'val_loss': loss, 'logits': logits, 'labels': label, 'img_id': img_id}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logits = torch.cat([x['logits'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])

        # Accuracy
        acc = self.acc_fn(logits, labels.squeeze())

        if self.experiment is not None:
            logs = {'val/loss': avg_loss.item(), 'val/acc': acc.item()}
            # Logging
            self.experiment.log_metrics(logs, epoch=self.epoch_num)

        # Save Weights
        if self.best_loss > avg_loss or self.best_acc < acc:
            self.best_loss = min(avg_loss.item(), self.best_loss)
            self.best_acc = max(acc.item(), self.best_acc)
            logs = {'val/best_loss': self.best_loss, 'val/best_acc': self.best_acc}
            self.experiment.log_parameters(logs)

            expname = self.cfg.data.exp_name
            filename = f'{expname}_seed_{self.cfg.data.seed}_fold_{self.cfg.train.fold}_epoch_{self.epoch_num}_loss_{avg_loss.item():.3f}_acc_{acc.item():.3f}.pth'
            torch.save(self.net.state_dict(), filename)
            if self.experiment is not None:
                self.experiment.log_model(name=filename, file_or_folder='./'+filename)
                os.remove(filename)

            # OOF
            oof = pd.DataFrame(logits.detach().cpu().numpy(), columns=[f'pred_label_{i}' for i in range(5)])
            oof.insert(0, 'label', labels.detach().cpu().numpy())
            ids = [x['img_id'] for x in outputs]
            ids = [list(x) for x in ids]
            ids = list(itertools.chain.from_iterable(ids))

            oof.insert(0, 'img_id', ids)
            oof_name = 'oof_' + f'_epoch_{self.epoch_num}' + f'_fold_{self.cfg.train.fold}' + '.csv'
            oof.to_csv(oof_name, index=False)
            self.experiment.log_asset(file_data=oof_name, file_name=oof_name)
            os.remove(oof_name)

        # Update Epoch Num
        self.epoch_num += 1

        return {'avg_val_loss': avg_loss}
