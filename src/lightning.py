import os
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import metrics

from .utils import CassavaDataset
from .cutmix import cutmix, CutMixCriterion, resizemix
from .mixup import mixup, MixupCriterion


class CassavaDataModule(pl.LightningDataModule):
    """
    DataModule for Cassava Competition
    """
    def __init__(self, data_dir, cfg, transform, cv,
                 use_merge=False, sample=False):
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
        self.sample = sample

    def prepare_data(self):
        # Prepare Data
        if self.use_merge:
            self.df = pd.read_csv(os.path.join(self.data_dir, 'merged.csv'))
        else:
            self.df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))

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
            img_id = None

        # Cutmix don't use on Last 5 Epoch
        th = 1.0 if self.cfg.train.epoch - self.current_epoch > 5 else 10000

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
