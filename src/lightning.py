from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from pytorch_lightning import metrics

from .utils import CassavaDataset

class CassavaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, cfg, transform, cv):
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
        self.test_dataset = CassavaDataset(self.data_dir, self.transform, phase='test', df=None)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.train.batch_size,
                          pin_memory=False,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.train.batch_size,
                          pin_memory=False,
                          shuffle=False)

    def test_dataloader(self):
        batch_size = min(len(self.test_dataset), self.cfg.train.batch_size)
        return DataLoader(self.test_dataset,
                          batch_size=batch_size,
                          pin_memory=False,
                          shuffle=False)


class CassavaLightningSystem(pl.LightningModule):
    def __init__(self, net, cfg, experiment=None):
        super(CassavaLightningSystem, self).__init__()
        self.net = net
        self.cfg = cfg
        self.experiment = experiment
        self.criterion = nn.CrossEntropyLoss()
        self.best_loss = 1e+9
        self.best_acc = None
        self.epoch_num = 0
        self.acc_fn = metrics.Accuracy()

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=self.cfg.train.lr, weight_decay=2e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.train.epoch, eta_min=0)

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

        if self.experiment is not None:
            logs = {'train/loss': loss.item()}
            self.experiment.log_metrics(logs, step=batch_idx)

        return {'loss': loss, 'logits': logits, 'labels': label}

    def validation_step(self, batch, batch_idx):
        loss, label, logits = self.step(batch)

        if self.experiment is not None:
            val_logs = {'val/loss': loss.item()}
            self.experiment.log_metrics(val_logs, step=batch_idx)

        return {'val_loss': loss, 'logits': logits, 'labels': label}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logits = torch.cat([x['logits'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])

        # Accuracy
        acc = self.acc_fn(logits, labels.squeeze())

        print(f'Epoch: {self.epoch_num}  Loss: {avg_loss.item():.4f}  Acc {acc.item():.4f}')

        if self.experiment is not None:
            logs = {'val/loss': avg_loss, 'val/acc': acc}
            # Logging
            self.experiment.log_metrics(logs, epoch=self.epoch_num)

        # Save Weights
        if self.best_loss > avg_loss:
            self.best_loss = avg_loss.item()
            self.best_acc = acc.item()
            expname = self.cfg.exp['exp_name']
            filename = f'{expname}_epoch_{self.epoch_num}_loss_{self.best_loss:.3f}_acc_{self.best_acc:.3f}.pth'
            torch.save(self.net.state_dict(), filename)
            if self.experiment is not None:
                self.experiment.log_model(name=filename, file_or_folder='./'+filename)
                os.remove(filename)

        # Update Epoch Num
        self.epoch_num += 1

        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        inp, img_id = batch
        out = self.forward(inp)
        logits = torch.sigmoid(out)

        return {'preds': logits, 'image_id': img_id}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs])
        _, preds = torch.max(preds, 1)

        preds = preds.detach().cpu().numpy()
        # [tuple, tuple]
        img_ids = [x['image_id'] for x in outputs]
        # [list, list]
        img_ids = [list(x) for x in img_ids]
        img_ids = list(itertools.chain.from_iterable(img_ids))

        self.sub = pd.DataFrame({
            'image_id': img_ids,
            'label': preds
        })

        return None