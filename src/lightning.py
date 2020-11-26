import os
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import metrics

from .utils import CassavaDataset

class CassavaDataModule(pl.LightningDataModule):
    """
    DataModule for Cassava Competition
    """
    def __init__(self, data_dir, cfg, transform, cv):
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

    def prepare_data(self):
        # Prepare Data
        self.df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))


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
                          pin_memory=True,
                          num_workers=self.cfg.train.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.train.batch_size,
                          pin_memory=True,
                          num_workers=self.cfg.train.num_workers,
                          shuffle=False)


class CassavaLightningSystem(pl.LightningModule):
    def __init__(self, net, cfg, optimizer, scheduler=None, experiment=None):
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
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = 1e+9
        self.best_acc = None
        self.epoch_num = 0
        self.acc_fn = metrics.Accuracy()

    def configure_optimizers(self):
        if self.scheduler is None:
            return [self.optimizer], []
        else:
            return [self.optimizer], [self.scheduler]

    def forward(self, x):
        return self.net(x)

    def step(self, batch):
        inp, label = batch
        out = self.forward(inp)
        loss = self.criterion(out, label.squeeze())

        return loss, label, torch.sigmoid(out)

    def training_step(self, batch, batch_idx):
        loss, label, logits = self.step(batch)

        return {'loss': loss, 'logits': logits, 'labels': label}

    def validation_step(self, batch, batch_idx):
        loss, label, logits = self.step(batch)

        return {'val_loss': loss, 'logits': logits, 'labels': label}

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
        if self.best_loss > avg_loss:
            self.best_loss = avg_loss.item()
            self.best_acc = acc.item()
            logs = {'val/best_loss': self.best_loss, 'val/best_acc': self.best_acc}
            self.experiment.log_parameters(logs)

            expname = self.cfg.data.exp_name
            filename = f'{expname}_epoch_{self.epoch_num}_loss_{self.best_loss:.3f}_acc_{self.best_acc:.3f}.pth'
            torch.save(self.net.state_dict(), filename)
            if self.experiment is not None:
                self.experiment.log_model(name=filename, file_or_folder='./'+filename)
                os.remove(filename)

        # Update Epoch Num
        self.epoch_num += 1

        return {'avg_val_loss': avg_loss}
