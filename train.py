import os
import hydra
from omegaconf import DictConfig
from comet_ml import Experiment
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.lightning import CassavaLightningSystem, CassavaDataModule
from src.models import enet
from src.utils import ImageTransform, seed_everything


@hydra.main('config.yml')
def main(cfg: DictConfig):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    # Config  -------------------------------------------------------------------
    data_dir = './data/'
    checkpoint_path = './checkpoints'

    seed_everything(cfg.data.seed)

    # Comet_ml
    experiment = Experiment(api_key=cfg.comet_ml.api_key,
                            project_name=cfg.comet_ml.project_name)


    # Data Module  ---------------------------------------------------------------
    transform = ImageTransform(img_size=cfg.data.img_size)
    cv = StratifiedKFold(n_splits=cfg.data.n_splits, shuffle=True, random_state=cfg.data.seed)
    dm = CassavaDataModule(data_dir, cfg, transform, cv)

    net = enet(model_type=cfg.train.model_type)

    model = CassavaLightningSystem(net, cfg, experiment)

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_top_k=1,
        monitor='avg_val_loss',
        verbose=False,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='avg_val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )

    trainer = Trainer(
        logger=False,
        max_epochs=cfg.train.epoch,
        gpus=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,  # Skip Sanity Check
    )

    # Train
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()