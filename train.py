import os
import glob
import hydra
from omegaconf import DictConfig
from comet_ml import Experiment

from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler
from timm.optim import RAdam
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.lightning import CassavaLightningSystem, CassavaDataModule
from src.augmentations import get_transforms
from src.models import Timm_model
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
    checkpoint_path = './checkpoints'
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
    dm.prepare_data()
    dm.setup()

    # Model  ----------------------------------------------------------------------
    net = Timm_model(cfg.train.model_type, pretrained=True)

    # Log Model Graph
    experiment.set_model_graph(str(net))

    # Loss fn  ---------------------------------------------------------------------
    criterion = get_loss_fn(cfg.train.loss_fn, smoothing=0.05)

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
    model = CassavaLightningSystem(net, cfg, criterion=criterion, optimizer=optimizer, scheduler=scheduler, experiment=experiment)

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
        patience=100,
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
        amp_level='O0',
        num_sanity_val_steps=0,  # Skip Sanity Check
        automatic_optimization=False if cfg.train.use_sam else True
    )

    # Train
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()