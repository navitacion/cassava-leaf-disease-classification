import os
import glob
import pandas as pd
import hydra
from omegaconf import DictConfig
from comet_ml import Experiment

from sklearn.model_selection import StratifiedKFold
import torch
from torch.optim import lr_scheduler
from timm.optim import RAdam
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.lightning import CassavaLightningSystem, CassavaDataModule
from src.augmentations import get_transforms
from src.models import Timm_model, Ensembler
from src.utils import seed_everything, CosineAnnealingWarmUpRestarts
from src.losses import get_loss_fn
from src.sam import SAM

DEBUG = False

@hydra.main('config.yml')
def main(cfg: DictConfig):
    print('Cassava Leaf Disease Classification')
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    # Config  -------------------------------------------------------------------
    data_dir = './input'
    seed_everything(cfg.data.seed)

    # Comet_ml
    experiment = Experiment(api_key=cfg.comet_ml.api_key,
                            project_name=cfg.comet_ml.project_name)

    # Log Parameters
    experiment.log_parameters(dict(cfg.data))
    experiment.log_parameters(dict(cfg.train))

    # Data Module  ---------------------------------------------------------------
    transform = get_transforms(transform_name=cfg.data.transform, img_size=cfg.data.img_size)
    cv = StratifiedKFold(n_splits=cfg.data.n_splits, shuffle=True, random_state=cfg.data.seed)
    dm = CassavaDataModule(data_dir, cfg, transform, cv,
                           use_merge=True,
                           drop_noise=cfg.data.drop_noise,
                           sample=DEBUG)

    # Model  ----------------------------------------------------------------------
    net = Timm_model(cfg.train.model_type, pretrained=True)
    # models = [cfg.train.model_type, 'vit_base_patch16_384']
    # weights = [0.6, 0.4]
    # net = Ensembler(models, weights, pretrained=True)

    # Log Model Graph
    experiment.set_model_graph(str(net))

    # Loss fn  ---------------------------------------------------------------------
    df = pd.read_csv('./input/merged.csv')
    weight = df['label'].value_counts().sort_index().tolist()
    weight = [w / len(df) for w in weight]
    weight = torch.tensor(weight).cuda()
    del df

    criterion = get_loss_fn(cfg.train.loss_fn, weight=weight, smoothing=0.05)

    # Optimizer, Scheduler  --------------------------------------------------------
    if cfg.train.use_sam:
        base_optimizer = RAdam
        optimizer = SAM(net.parameters(), base_optimizer, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    else:
        optimizer = RAdam(net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    if cfg.train.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epoch, eta_min=0)
    elif cfg.train.scheduler == 'cosine-warmup':
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=cfg.train.epoch, T_up=5, eta_max=cfg.train.lr * 10)

    # Lightning Module  -------------------------------------------------------------
    model = CassavaLightningSystem(net, cfg,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   experiment=experiment)

    # Trainer  -------------------------------------------------------------------------
    trainer = Trainer(
        logger=False,
        max_epochs=cfg.train.epoch,
        gpus=-1,
        amp_backend='apex',
        amp_level='O2',
        num_sanity_val_steps=0,  # Skip Sanity Check
        automatic_optimization=False if cfg.train.use_sam else True,
        # resume_from_checkpoint='./checkpoints/epoch=3-step=14047.ckpt'
    )

    # Train
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()