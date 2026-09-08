import os
import sys
import yaml
import glob
import torch
import pprint
from munch import munchify
from models import VisDynamicsModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)
        
class LossCurvePlotter(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.rec_losses = []
        self.epoch_numbers = []

    def on_validation_end(self, trainer, pl_module):
        # Get current epoch and logged metrics
        epoch = trainer.current_epoch
        # The metrics may be logged as tensors; extract their values
        if 'train/loss_step' in trainer.callback_metrics:
            train_loss = trainer.callback_metrics['train/loss_step']
            self.train_losses.append(train_loss.item())
        else:
            self.train_losses.append(float('nan'))
        if 'train/rec_loss_step' in trainer.callback_metrics:
            rec_loss = trainer.callback_metrics['train/rec_loss_step']
            self.rec_losses.append(rec_loss.item())
        else:
            self.rec_losses.append(float('nan'))
        self.epoch_numbers.append(epoch)

    def on_train_end(self, trainer, pl_module):
        # Plot the loss curve and save to file
        plt.figure()
        plt.plot(self.epoch_numbers, self.train_losses, label="Train Loss")
        plt.plot(self.epoch_numbers, self.rec_losses, label="Rec Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        # Save in the log directory; you can change the filename as needed.
        plot_path = os.path.join(pl_module.hparams.log_dir, f"{pl_module.hparams.dataset}_loss_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print("Saved loss curve to:", plot_path)

def main():
    config_filepath = str(sys.argv[1])
    high_dim_ckpt_dir = str(sys.argv[2])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)            
    log_dir_name = '_'.join([cfg.dataset, cfg.model_name, str(cfg.seed)])
    log_dir = os.path.join(cfg.log_dir, log_dir_name)
                            
    model = VisDynamicsModel(
        beta=cfg.beta,
        lda1=cfg.lda1,
        lda2=cfg.lda2,
        lda3=cfg.lda3,
        lr=cfg.lr,
        seed=cfg.seed,
        if_cuda=cfg.if_cuda,
        if_test=False,
        gamma=cfg.gamma,
        log_dir=log_dir,
        train_batch=cfg.train_batch,
        val_batch=cfg.val_batch,
        test_batch=cfg.test_batch,
        num_workers=cfg.num_workers,
        model_name=cfg.model_name,
        data_filepath=cfg.data_filepath,
        dataset=cfg.dataset,
        num_frames=cfg.num_frames,
        lr_schedule=cfg.lr_schedule,
        high_dim_ckpt_dir=high_dim_ckpt_dir
    )

    # define callback for selecting checkpoints during training
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir + '/lightning_logs/checkpoints/',
        filename='epoch={epoch}-val_loss={val/loss:.4f}',
        auto_insert_metric_name=False,
        verbose=True,
        monitor='val/loss',
        mode='min',
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir + '/lightning_logs/last_checkpoints/',
        filename='epoch={epoch}-val_loss={val/loss:.4f}',
        auto_insert_metric_name=False,
    )
    loss_plotter = LossCurvePlotter()
    
    trainer = Trainer(
        accelerator='gpu',
        devices=cfg.num_gpus,
        # num_nodes=cfg.num_gpus,
        max_epochs=cfg.epochs,
        deterministic=True,
        strategy='ddp',  # or 'ddp' if appropriate
        default_root_dir=log_dir,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_checkpointing=True,
        accumulate_grad_batches=2,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback, loss_plotter],
        precision=16
    )



    trainer.fit(model)

if __name__ == '__main__':
    main()
