import os
import cv2
import glob
import hydra
from omegaconf import DictConfig
from comet_ml import Experiment
from sklearn.model_selection import StratifiedKFold
from torch import optim
from torch.optim import lr_scheduler
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.lightning import CassavaLightningSystem, CassavaDataModule
from src.models import enet, Timm_model
from src.utils import seed_everything, CosineAnnealingWarmUpRestarts


# Image Augmentations
class ImageTransform:
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = {
            'train': albu.Compose([
                albu.RandomShadow(p=0.5),
                albu.RandomResizedCrop(img_size, img_size, interpolation=cv2.INTER_AREA),
                albu.ColorJitter(p=0.5),
                albu.CLAHE(p=0.5),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Transpose(p=0.5),
                albu.ShiftScaleRotate(p=0.5),
                albu.OneOf([
                    albu.Blur(p=1.0),
                albu.GaussianBlur(p=1.0)
                ], p=0.5),
                albu.CoarseDropout(max_height=15, max_width=15, min_holes=3, p=0.5),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0),

            'val': albu.Compose([
                albu.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0)
        }

    def __call__(self, img, phase='train'):
        augmented = self.transform[phase](image=img)
        augmented = augmented['image']

        return augmented


@hydra.main('config.yml')
def main(cfg: DictConfig):
    print('Cassava Leaf Disease Classification')
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    # Config  -------------------------------------------------------------------
    data_dir = './input'
    checkpoint_path = './checkpoints'
    seed_everything(cfg.data.seed)

    # Comet_ml
    experiment = Experiment(api_key=cfg.comet_ml.api_key,
                            project_name=cfg.comet_ml.project_name)

    # Log Parameters
    experiment.log_parameters(dict(cfg.data))
    experiment.log_parameters(dict(cfg.train))

    # Data Module  ---------------------------------------------------------------
    transform = ImageTransform(img_size=cfg.data.img_size)
    cv = StratifiedKFold(n_splits=cfg.data.n_splits, shuffle=True, random_state=cfg.data.seed)
    dm = CassavaDataModule(data_dir, cfg, transform, cv)
    dm.prepare_data()
    dm.setup()

    # Model  ----------------------------------------------------------------------
    net = Timm_model(cfg.train.model_type, pretrained=True)
    # Log Model Graph
    experiment.set_model_graph(str(net))

    # Optimizer, Scheduler  --------------------------------------------------------
    optimizer = optim.AdamW(net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    if cfg.train.scheduler == 'onecycle':
        steps_per_epoch = (len(glob.glob(os.path.join(data_dir, 'train_images', '*.jpg'))) // cfg.train.batch_size) + 1
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.train.lr, epochs=cfg.train.epoch, steps_per_epoch=steps_per_epoch)
    elif cfg.train.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epoch, eta_min=0)
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=cfg.train.epoch, T_up=5, eta_max=cfg.train.lr * 10)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epoch, eta_min=0)

    # Lightning Module  -------------------------------------------------------------
    model = CassavaLightningSystem(net, cfg, optimizer=optimizer, scheduler=scheduler, experiment=experiment)

    # Callbacks  --------------------------------------------------------------------
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_top_k=1,
        monitor='avg_val_loss',
        verbose=False,
        mode='min',
        prefix=cfg.data.exp_name + '_'
    )

    early_stop_callback = EarlyStopping(
        monitor='avg_val_loss',
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode='min'
    )

    # Trainer  -------------------------------------------------------------------------
    trainer = Trainer(
        logger=False,
        max_epochs=cfg.train.epoch,
        gpus=-1,
        callbacks=[checkpoint_callback, early_stop_callback],
        amp_backend='apex',
        amp_level='O2',
        # num_sanity_val_steps=0,  # Skip Sanity Check
    )

    # Train
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()