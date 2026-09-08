import os
import glob
import time
import torch
import shutil
import numpy as np
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import imageio
from dataset import NeuralPhysDataset, NeuralPhysRefineDataset
from model_utils import (
    EncoderDecoder64x1x1,
    EncoderDecoderDynamicsNetwork,
    RefineModel,
    RefineDynamicsModel,
    ViTEncoderDecoder,  # <-- New ViT-based encoder-decoder
    ViTTimeSformerEncoderDecoder,
)
def generate_video_from_predictions(predictions_path, output_video_path, fps=10):
    """
    Generate a video from prediction images stored in predictions_path.

    Args:
        predictions_path (str): The directory that contains your prediction images.
        output_video_path (str): The file path where the video will be saved.
        fps (int): Frames per second for the output video.
    """
    # Get a sorted list of all image files in predictions_path.
    # This assumes that the filenames are such that sorting them will result in chronological order.
    img_files = sorted([os.path.join(predictions_path, f)
                        for f in os.listdir(predictions_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not img_files:
        print("No image files found in", predictions_path)
        return

    writer = imageio.get_writer(output_video_path, fps=fps)
    for img_file in img_files:
        frame = imageio.imread(img_file)
        writer.append_data(frame)
    writer.close()
    print("Video saved to:", output_video_path)
def mkdir(folder):
    if os.path.exists(folder):
        # If the folder exists, rename it by appending the current timestamp.
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        new_folder = f"{folder}_{timestamp}"
        shutil.move(folder, new_folder)
        print(f"Existing folder '{folder}' has been renamed to '{new_folder}'.")
    os.makedirs(folder, exist_ok=True)

def rename_ckpt_for_multi_models(ckpt):
    renamed_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if 'dynamics' in k:
            continue
        if 'high_dim_model' in k:
            continue
        else:
            name = k.replace('model.', '')
        renamed_state_dict[name] = v
    return renamed_state_dict

def load_checkpoint(filepath, model):
    checkpoint_filepath = glob.glob(os.path.join(filepath, '*.ckpt'))[0]
    print('Loading', checkpoint_filepath)
    ckpt = torch.load(checkpoint_filepath)
    ckpt = rename_ckpt_for_multi_models(ckpt)
    model.load_state_dict(ckpt)

class VisDynamicsModel(pl.LightningModule):

    def __init__(
        self,
        lda1: float=1.0,
        lda2: float=1.0,
        lda3: float=1.0,
        beta: float=1.0,
        lr: float=1e-4,
        seed: int=1,
        if_cuda: bool=True,
        if_test: bool=False,
        gamma: float=0.5,
        log_dir: str='logs',
        train_batch: int=512,
        val_batch: int=256,
        test_batch: int=256,
        num_workers: int=8,
        model_name: str='encoder-decoder-64',
        data_filepath: str='data',
        dataset: str='single_pendulum',
        num_frames: int=60,
        lr_schedule: list=[],
        high_dim_ckpt_dir: str='',
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.global_iter = 0
        assert self.hparams.if_cuda
        self.kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True}
        # create visualization saving folder if testing
        self.pred_log_dir = os.path.join(self.hparams.log_dir, 'predictions')
        self.var_log_dir = os.path.join(self.hparams.log_dir, 'variables')
        if not self.hparams.if_test:
            mkdir(self.pred_log_dir)
            mkdir(self.var_log_dir)
        self._build_model()

    def _build_model(self):
        if self.hparams.model_name == 'encoder-decoder-64':
            self.model = EncoderDecoder64x1x1(in_channels=3)
            self.dynamics_model = EncoderDecoderDynamicsNetwork(in_channels=64)

        elif self.hparams.model_name == 'refine-64':
            self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)
            load_checkpoint(self.hparams.high_dim_ckpt_dir, self.high_dim_model)
            for name, param in self.high_dim_model.named_parameters():
                param.requires_grad = False
            if self.hparams.dataset == 'reaction_diffusion':
                self.model = RefineModel(in_channels=64, intrinsic_dim=2)
                self.dynamics_model = RefineDynamicsModel(in_channels=2)
            elif self.hparams.dataset == 'circular_motion':
                self.model = RefineModel(in_channels=64, intrinsic_dim=2)
                self.dynamics_model = RefineDynamicsModel(in_channels=2)
            elif self.hparams.dataset == 'single_pendulum':
                self.model = RefineModel(in_channels=64, intrinsic_dim=2)
                self.dynamics_model = RefineDynamicsModel(in_channels=2)
            elif self.hparams.dataset == 'double_pendulum':
                self.model = RefineModel(in_channels=64, intrinsic_dim=4)
                self.dynamics_model = RefineDynamicsModel(in_channels=4)
            elif self.hparams.dataset == 'elastic_pendulum':
                self.model = RefineModel(in_channels=64, intrinsic_dim=6)
                self.dynamics_model = RefineDynamicsModel(in_channels=6)
            elif self.hparams.dataset == 'lava_lamp':
                self.model = RefineModel(in_channels=64, intrinsic_dim=4)
                self.dynamics_model = RefineDynamicsModel(in_channels=4)
            elif self.hparams.dataset == 'fire_flame':
                self.model = RefineModel(in_channels=64, intrinsic_dim=8)
                self.dynamics_model = RefineDynamicsModel(in_channels=8)
            elif self.hparams.dataset == 'air_dancer':
                self.model = RefineModel(in_channels=64, intrinsic_dim=4)
                self.dynamics_model = RefineDynamicsModel(in_channels=4)
            elif self.hparams.dataset == 'swing_stick':
                self.model = RefineModel(in_channels=64, intrinsic_dim=4)
                self.dynamics_model = RefineDynamicsModel(in_channels=4)
        elif self.hparams.model_name == 'vit-tide':
            # New branch using the ViT-based encoder
            # if self.hparams.dataset == 'circular_motion':
            #     self.model = ViTEncoderDecoder(img_size=(128,256), patch_size=8, in_channels=3, embed_dim=64)
            # else:
            #     self.model = ViTEncoderDecoder(img_size=(128,256), patch_size=16, in_channels=3, embed_dim=64)
            # self.dynamics_model = EncoderDecoderDynamicsNetwork(in_channels=64)
            self.model = ViTTimeSformerEncoderDecoder(
                img_size=(128, 256),
                patch_size=16,
                in_channels=3,
                embed_dim=64,
                spatial_depth=6,
                spatial_heads=8,
                temporal_depth=4,
                temporal_heads=8,
                causal=True,
            )
            self.dynamics_model = None
        else:
            raise ValueError(f"Unknown model_name: {self.hparams.model_name}")
    def forward(self, x, reconstructed_latent=None):
        # For simplicity, we assume that the underlying model’s forward method does
        # the appropriate computation for the variant in use (e.g., "vit-tide").
        # You can add conditions if necessary.
        return self.model(x, reconstructed_latent)

    def encoder_decoder64_loss(
        self, 
        target, 
        output, 
        next_output, 
        latent, 
        next_latent, 
        mu, 
        logvar
    ):
        B, F = target.shape[0], target.shape[1]
        output = torch.reshape(output, target.shape)
        next_output = torch.reshape(next_output, target.shape)
        latent = torch.reshape(latent, (B, F, 64))
        next_latent = torch.reshape(next_latent, (B, F, 64))
        mu = torch.reshape(mu, (B, F, 64))
        logvar = torch.reshape(logvar, (B, F, 64))

        mse_loss = torch.nn.MSELoss(reduction='sum')
        # reconstruct current and next state
        cur_frame_loss = mse_loss(output, target) / (B*F)
        next_frame_loss = mse_loss(next_output[:, :-1], target[:, 1:]) / (B*(F-1))
        rec_loss = cur_frame_loss + next_frame_loss
        # reconstruct next 64-dim latent vector
        nll_loss = logvar[:, 1:] + (next_latent[:, :-1] - mu[:, 1:]) ** 2 / logvar[:, 1:].exp()
        latent_rec_loss = torch.sum(nll_loss) / (2*B*(F-1))
        # KL divergence and regularization terms
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (B*F)
        reg_loss = self.smoothness_loss(mu)

        return (
            rec_loss + self.hparams.lda1 * reg_loss + self.hparams.lda2 * latent_rec_loss + self.hparams.beta * kl_loss,
            rec_loss, 
            reg_loss,
            latent_rec_loss,
            kl_loss, 
        )
    
    def refine_loss(
        self, 
        target, 
        output, 
        next_output, 
        mu, 
        logvar, 
        reconstructed_latent, 
        next_reconstructed_latent, 
        latent_mu,
        latent_logvar,
        latent_latent, 
        next_latent_latent,
        is_test=False,
    ):
        B, F = target.shape[0], target.shape[1]
        output = torch.reshape(output, target.shape)
        next_output = torch.reshape(next_output, target.shape)
        mu = torch.reshape(mu, (B, F, 64))
        logvar = torch.reshape(logvar, (B, F, 64))
        reconstructed_latent = torch.reshape(reconstructed_latent, mu.shape)
        next_reconstructed_latent = torch.reshape(next_reconstructed_latent, mu.shape)
        latent_mu = torch.reshape(latent_mu, (B, F, latent_mu.shape[-1]))
        latent_logvar = torch.reshape(latent_logvar, (B, F, latent_logvar.shape[-1]))
        latent_latent = torch.reshape(latent_latent, (B, F, latent_latent.shape[-1]))
        next_latent_latent = torch.reshape(next_latent_latent, (B, F, next_latent_latent.shape[-1]))

        mse_loss = torch.nn.MSELoss(reduction='sum')
        # reconstruct current and next state
        cur_frame_loss = mse_loss(output, target) / (B*F)
        next_frame_loss = mse_loss(next_output[:, :-1], target[:, 1:]) / (B*(F-1))
        rec_loss = cur_frame_loss + next_frame_loss
        # reconstruct next state variables
        latent_nll_loss = latent_logvar[:, 1:] + (next_latent_latent[:, :-1] - latent_mu[:, 1:]) ** 2 / latent_logvar[:, 1:].exp()
        latent_rec_loss = torch.sum(latent_nll_loss) / (2*B*F)
        # reconstruct 64-dim latent vector
        latent64_nll_loss = logvar + (reconstructed_latent - mu) ** 2 / logvar.exp()
        latent64_rec_loss = torch.sum(latent64_nll_loss) / (2*B*F)
        latent64_next_nll_loss = logvar[:, 1:] + (next_reconstructed_latent[:, :-1] - mu[:, 1:]) ** 2 / logvar[:, 1:].exp()
        latent64_rec_loss += torch.sum(latent64_next_nll_loss) / (2*B*(F-1))
        # KL divergence and regularization terms
        kl_loss = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp()) / (B*F)
        reg_loss = self.smoothness_loss(latent_mu)

        if is_test:
            return (
                rec_loss + self.hparams.lda1 * reg_loss + self.hparams.lda2 * latent64_rec_loss + self.hparams.lda3 * latent_rec_loss + self.hparams.beta * kl_loss,
                (rec_loss, cur_frame_loss, next_frame_loss), 
                reg_loss,
                latent64_rec_loss,
                latent_rec_loss, 
                kl_loss,
            )
        return (
            rec_loss + self.hparams.lda1 * reg_loss + self.hparams.lda2 * latent64_rec_loss + self.hparams.lda3 * latent_rec_loss + self.hparams.beta * kl_loss,
            rec_loss, 
            reg_loss,
            latent64_rec_loss,
            latent_rec_loss, 
            kl_loss,
        )

    def smoothness_loss(self, latent):
        loss = 0
        derivs = 4
        maxs = torch.amax(latent, dim=1, keepdim=True)
        mins = torch.amin(latent, dim=1, keepdim=True)
        scaled_latent = latent / (maxs - mins + 1e-12)
        last_deriv = scaled_latent
        for k in range(derivs):
            deriv = last_deriv[:, 1:] - last_deriv[:, :-1]
            deriv_loss = torch.mean(torch.sum(torch.abs(deriv), dim=1))
            loss += (5.0**(k+1)) * deriv_loss
            last_deriv = deriv
        return loss

    def training_step(self, batch, batch_idx):
        data, target, filepath = batch
        if self.hparams.model_name == 'encoder-decoder-64':
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            output, latent, mu, logvar = self.model(data, None)
            next_latent = self.dynamics_model(mu)
            next_output = self.model.decoder(next_latent)

            losses = self.encoder_decoder64_loss(target, output, next_output, latent, next_latent, mu, logvar)
            train_loss, train_rec_loss, train_reg_loss, train_latent_rec_loss, train_kl_loss = losses
            self.log('train/loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/rec_loss', train_rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/reg_loss', train_reg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/latent_rec_loss', train_latent_rec_loss, on_step=True, on_epoch=True, prog_bar=True,
                     logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/kl_loss', train_kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.global_iter += 1
            
        elif self.hparams.model_name == 'refine-64':
            self.high_dim_model.eval()
            mu, logvar = data[:, :, :, 0], data[:, :, :, 1] 
            mu = torch.reshape(mu, (-1, mu.shape[-1]))
            reconstructed_latent, latent_latent, latent_mu, latent_logvar = self.model(mu)
            next_latent_latent = self.dynamics_model(latent_mu)
            next_reconstructed_latent = self.model.decoder(next_latent_latent)
            next_output = self.high_dim_model.decoder(next_reconstructed_latent)
            output = self.high_dim_model.decoder(reconstructed_latent)

            losses = self.refine_loss(target, output, next_output, mu, logvar,
                                       reconstructed_latent, next_reconstructed_latent,
                                       latent_mu, latent_logvar, latent_latent, next_latent_latent)
            train_loss, train_rec_loss, train_reg_loss, train_latent64_rec_loss, train_latent_rec_loss, train_kl_loss = losses 
            self.log('train/loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/rec_loss', train_rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/reg_loss', train_reg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/latent64_rec_loss', train_latent64_rec_loss, on_step=True, on_epoch=True, prog_bar=True,
                     logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/latent_rec_loss', train_latent_rec_loss, on_step=True, on_epoch=True, prog_bar=True,
                     logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/kl_loss', train_kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
        elif self.hparams.model_name == 'vit-tide':
            # DO NOT flatten; keep [B, F, C, H, W]
            output, latent_seq, mu_seq, logvar_seq = self.model(data, None)

            # Reuse existing loss: feed output twice and latents twice
            losses = self.encoder_decoder64_loss(
            target, output, output, latent_seq, latent_seq, mu_seq, logvar_seq
            )
            train_loss, train_rec_loss, train_reg_loss, train_latent_rec_loss, train_kl_loss = losses

            self.log('train/loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/rec_loss', train_rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/reg_loss', train_reg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/latent_rec_loss', train_latent_rec_loss, on_step=True, on_epoch=True, prog_bar=True,
             logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('train/kl_loss', train_kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)
            self.global_iter += 1

        # elif self.hparams.model_name == 'vit-tide':
        #     data = torch.reshape(data, (-1, *data.shape[-3:]))
        #     output, latent, mu, logvar = self.model(data, None)
        #     next_latent = self.dynamics_model(mu)
        #     next_output = self.model.decoder(next_latent)

        #     losses = self.encoder_decoder64_loss(target, output, next_output, latent, next_latent, mu, logvar)
        #     train_loss, train_rec_loss, train_reg_loss, train_latent_rec_loss, train_kl_loss = losses
            # if hasattr(self.hparams, 'kl_warmup_epochs') and self.hparams.kl_warmup_epochs > 0:
            #     effective_beta = self.hparams.beta * min(1.0, (self.current_epoch + 1) / self.hparams.kl_warmup_epochs)
            # else:
            #     effective_beta = self.hparams.beta

            # # Recombine loss components
            # total_loss = train_rec_loss + self.hparams.lda1 * train_reg_loss + self.hparams.lda2 * train_latent_rec_loss + effective_beta *train_kl_loss

            # # Debug logging: print loss breakdown for every 5 batches
            # if batch_idx % 5 == 0:
            #     debug_message = (f"Epoch {self.current_epoch}, Batch {batch_idx}: "
            #              f"Rec={train_rec_loss.item():.2f}, Reg={train_reg_loss.item():.2f}, "
            #              f"LatentRec={train_latent_rec_loss.item():.2f}, KL={train_kl_loss.item():.2f}, "
            #              f"EffectiveBeta={effective_beta:.3f}")
            #     self.logger.experiment.add_text("Loss Breakdown", debug_message, self.global_iter)
            #     (debug_message)  # Also print to stdout for immediate feedback


            # Recompute total loss using the effective KL weight
            # train_loss = train_rec_loss + self.hparams.lda1 * train_reg_loss + self.hparams.lda2 * train_latent_rec_loss + effective_beta * train_kl_loss
            # self.log('train/loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
            #          batch_size=self.hparams.train_batch, sync_dist=True)
            # self.log('train/rec_loss', train_rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
            #          batch_size=self.hparams.train_batch, sync_dist=True)
            # self.log('train/reg_loss', train_reg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
            #          batch_size=self.hparams.train_batch, sync_dist=True)
            # self.log('train/latent_rec_loss', train_latent_rec_loss, on_step=True, on_epoch=True, prog_bar=True,
            #          logger=True, batch_size=self.hparams.train_batch, sync_dist=True)
            # self.log('train/kl_loss', train_kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
            #          batch_size=self.hparams.train_batch, sync_dist=True)
            # self.global_iter += 1
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        data, target, filepath = batch
        if self.hparams.model_name == 'encoder-decoder-64':
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            output, latent, mu, logvar = self.model(data, None)
            next_latent = self.dynamics_model(mu)
            next_output = self.model.decoder(next_latent)

            losses = self.encoder_decoder64_loss(target, output, next_output, latent, next_latent, mu, logvar)
            val_loss, val_rec_loss, val_reg_loss, val_latent_rec_loss, val_kl_loss = losses
            self.log('val/loss', val_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/rec_loss', val_rec_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/reg_loss', val_reg_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/latent_rec_loss', val_latent_rec_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/kl_loss', val_kl_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.global_iter += 1
        
        elif self.hparams.model_name == 'refine-64':
            self.high_dim_model.eval()
            mu, logvar = data[:, :, :, 0], data[:, :, :, 1] 
            mu = torch.reshape(mu, (-1, mu.shape[-1]))
            reconstructed_latent, latent_latent, latent_mu, latent_logvar = self.model(mu)
            next_latent_latent = self.dynamics_model(latent_mu)
            next_reconstructed_latent = self.model.decoder(next_latent_latent)
            next_output = self.high_dim_model.decoder(next_reconstructed_latent)
            output = self.high_dim_model.decoder(reconstructed_latent)

            losses = self.refine_loss(target, output, next_output, mu, logvar,
                                       reconstructed_latent, next_reconstructed_latent,
                                       latent_mu, latent_logvar, latent_latent, next_latent_latent)
            val_loss, val_rec_loss, val_reg_loss, val_latent64_rec_loss, val_latent_rec_loss, val_kl_loss = losses 
            self.log('val/loss', val_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/rec_loss', val_rec_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/reg_loss', val_reg_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/latent64_rec_loss', val_latent64_rec_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/latent_rec_loss', val_latent_rec_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/kl_loss', val_kl_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
        elif self.hparams.model_name == 'vit-tide':
            output, latent_seq, mu_seq, logvar_seq = self.model(data, None)
            losses = self.encoder_decoder64_loss(
                target, output, output, latent_seq, latent_seq, mu_seq, logvar_seq
            )
            val_loss, val_rec_loss, val_reg_loss, val_latent_rec_loss, val_kl_loss = losses

            self.log('val/loss', val_loss, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/rec_loss', val_rec_loss, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/reg_loss', val_reg_loss, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/latent_rec_loss', val_latent_rec_loss, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('val/kl_loss', val_kl_loss, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)
            self.global_iter += 1

        # elif self.hparams.model_name == 'vit-tide':
        #     data = torch.reshape(data, (-1, *data.shape[-3:]))
        #     output, latent, mu, logvar = self.model(data, None)
        #     next_latent = self.dynamics_model(mu)
        #     next_output = self.model.decoder(next_latent)

        #     losses = self.encoder_decoder64_loss(target, output, next_output, latent, next_latent, mu, logvar)
        #     val_loss, val_rec_loss, val_reg_loss, val_latent_rec_loss, val_kl_loss = losses
        #     self.log('val/loss', val_loss, on_epoch=True, prog_bar=True, logger=True,
        #              batch_size=self.hparams.train_batch, sync_dist=True)
        #     self.log('val/rec_loss', val_rec_loss, on_epoch=True, prog_bar=True, logger=True,
        #              batch_size=self.hparams.train_batch, sync_dist=True)
        #     self.log('val/reg_loss', val_reg_loss, on_epoch=True, prog_bar=True, logger=True,
        #              batch_size=self.hparams.train_batch, sync_dist=True)
        #     self.log('val/latent_rec_loss', val_latent_rec_loss, on_epoch=True, prog_bar=True, logger=True,
        #              batch_size=self.hparams.train_batch, sync_dist=True)
        #     self.log('val/kl_loss', val_kl_loss, on_epoch=True, prog_bar=True, logger=True,
        #              batch_size=self.hparams.train_batch, sync_dist=True)
        #     self.global_iter += 1
        
        return val_loss

    def test_step(self, batch, batch_idx):
        
        if self.hparams.model_name == 'encoder-decoder-64':
            data, target, filepath = batch
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            output, latent, mu, logvar = self.model(data, None)
            next_latent = self.dynamics_model(mu)
            next_output = self.model.decoder(next_latent)

            losses = self.encoder_decoder64_loss(target, output, next_output, latent, next_latent, mu, logvar)
            test_loss, test_rec_loss, test_reg_loss, test_latent_rec_loss, test_kl_loss = losses
            self.log('test/loss', test_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/rec_loss', test_rec_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/reg_loss', test_reg_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/latent_rec_loss', test_latent_rec_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/kl_loss', test_kl_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            
            target = torch.reshape(target, (-1, *target.shape[-3:]))
            filepath = np.transpose(filepath).flatten()
            self.all_filepaths.extend(filepath)
            for idx in range(data.shape[0]):
                comparison = torch.cat([
                    target[idx, :, :, :128].unsqueeze(0),
                    target[idx, :, :, 128:].unsqueeze(0),
                    output[idx, :, :, :128].unsqueeze(0),
                    output[idx, :, :, 128:].unsqueeze(0),
                    next_output[idx, :, :, :128].unsqueeze(0),
                    next_output[idx, :, :, 128:].unsqueeze(0)
                ])
                save_image(comparison.cpu(), os.path.join(self.pred_log_dir, filepath[idx]), nrow=1)
                latent_tmp = latent[idx].view(1, -1)[0].cpu().detach().numpy()
                self.all_latents.append(latent_tmp)
                mu_tmp = mu[idx].view(1, -1)[0].cpu().detach().numpy()
                self.all_mus.append(mu_tmp)
                logvar_tmp = logvar[idx].view(1, -1)[0].cpu().detach().numpy()
                self.all_logvars.append(logvar_tmp)
                next_latent_tmp = next_latent[idx].view(1, -1)[0].cpu().detach().numpy()
                self.all_next_latents.append(next_latent_tmp)

        elif self.hparams.model_name  == 'refine-64':
            self.high_dim_model.eval()
            data, target, filepath = batch
            data = torch.reshape(data, (-1, *data.shape[-3:]))
            _, _, mu, logvar = self.high_dim_model(data, None)
            reconstructed_latent, latent_latent, latent_mu, latent_logvar = self.model(mu)
            next_latent_latent = self.dynamics_model(latent_mu)
            next_reconstructed_latent = self.model.decoder(next_latent_latent)
            next_output = self.high_dim_model.decoder(next_reconstructed_latent)
            output = self.high_dim_model.decoder(reconstructed_latent)

            losses = self.refine_loss(target, output, next_output, mu, logvar,
                                       reconstructed_latent, next_reconstructed_latent,
                                       latent_mu, latent_logvar, latent_latent, next_latent_latent, is_test=True)
            test_loss, test_rec_loss, test_reg_loss, test_latent64_rec_loss, test_latent_rec_loss, test_kl_loss = losses
            test_rec_loss, cur_frame_loss, next_frame_loss = test_rec_loss
            self.log('test/loss', test_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/rec_loss', test_rec_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/cur_frame_loss', cur_frame_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/next_frame_loss', next_frame_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/reg_loss', test_reg_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/latent64_rec_loss', test_latent64_rec_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/latent_rec_loss', test_latent_rec_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/kl_loss', test_kl_loss, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.hparams.train_batch, sync_dist=True)
            
            target = torch.reshape(target, (-1, *target.shape[-3:]))
            filepath = np.transpose(filepath).flatten()
            self.all_filepaths.extend(filepath)
            for idx in range(data.shape[0]):
                comparison = torch.cat([
                    target[idx, :, :, :128].unsqueeze(0),
                    target[idx, :, :, 128:].unsqueeze(0),
                    output[idx, :, :, :128].unsqueeze(0),
                    output[idx, :, :, 128:].unsqueeze(0),
                    next_output[idx, :, :, :128].unsqueeze(0),
                    next_output[idx, :, :, 128:].unsqueeze(0)
                ])
                save_image(comparison.cpu(), os.path.join(self.pred_log_dir, filepath[idx]), nrow=1)
                latent_tmp = mu[idx].view(1, -1)[0].cpu().detach().numpy()
                self.all_latents.append(latent_tmp)
        elif self.hparams.model_name == 'vit-tide':
            data, target, filepath = batch
            output, latent_seq, mu_seq, logvar_seq = self.model(data, None)

            losses = self.encoder_decoder64_loss(
                target, output, output, latent_seq, latent_seq, mu_seq, logvar_seq
            )
            test_loss, test_rec_loss, test_reg_loss, test_latent_rec_loss, test_kl_loss = losses

            self.log('test/loss', test_loss, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/rec_loss', test_rec_loss, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/reg_loss', test_reg_loss, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/latent_rec_loss', test_latent_rec_loss, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)
            self.log('test/kl_loss', test_kl_loss, on_epoch=True, prog_bar=True, logger=True,
             batch_size=self.hparams.train_batch, sync_dist=True)

            # ---- saving visuals/latents (flatten sequence for existing saver) ----
            B, F = data.shape[:2]
            tgt_flat = target.view(-1, *target.shape[-3:])
            out_flat = output.view(-1, *output.shape[-3:])
            filepath = np.transpose(filepath).flatten()
            self.all_filepaths.extend(filepath)
            for idx in range(tgt_flat.shape[0]):
                comparison = torch.cat([
                    tgt_flat[idx, :, :, :128].unsqueeze(0),
                    tgt_flat[idx, :, :, 128:].unsqueeze(0),
                    out_flat[idx, :, :, :128].unsqueeze(0),
                    out_flat[idx, :, :, 128:].unsqueeze(0),
                    out_flat[idx, :, :, :128].unsqueeze(0),   # using output as "next_output"
                    out_flat[idx, :, :, 128:].unsqueeze(0)
                ])
                save_image(comparison.cpu(), os.path.join(self.pred_log_dir, filepath[idx]), nrow=1)

            lat_flat  = latent_seq.reshape(-1, latent_seq.shape[-1]).detach().cpu().numpy()
            mu_flat   = mu_seq.reshape(-1, mu_seq.shape[-1]).detach().cpu().numpy()
            logv_flat = logvar_seq.reshape(-1, logvar_seq.shape[-1]).detach().cpu().numpy()
            for i in range(lat_flat.shape[0]):
                self.all_latents.append(lat_flat[i])
                self.all_mus.append(mu_flat[i])
                self.all_logvars.append(logv_flat[i])

            #return test_loss

        # elif self.hparams.model_name == 'vit-tide':
        #     data, target, filepath = batch
        #     data = torch.reshape(data, (-1, *data.shape[-3:]))
        #     output, latent, mu, logvar = self.model(data, None)
        #     next_latent = self.dynamics_model(mu)
        #     next_output = self.model.decoder(next_latent)

        #     losses = self.encoder_decoder64_loss(target, output, next_output, latent, next_latent, mu, logvar)
        #     test_loss, test_rec_loss, test_reg_loss, test_latent_rec_loss, test_kl_loss = losses
        #     self.log('test/loss', test_loss, on_epoch=True, prog_bar=True, logger=True,
        #              batch_size=self.hparams.train_batch, sync_dist=True)
        #     self.log('test/rec_loss', test_rec_loss, on_epoch=True, prog_bar=True, logger=True,
        #              batch_size=self.hparams.train_batch, sync_dist=True)
        #     self.log('test/reg_loss', test_reg_loss, on_epoch=True, prog_bar=True, logger=True,
        #              batch_size=self.hparams.train_batch, sync_dist=True)
        #     self.log('test/latent_rec_loss', test_latent_rec_loss, on_epoch=True, prog_bar=True, logger=True,
        #              batch_size=self.hparams.train_batch, sync_dist=True)
        #     self.log('test/kl_loss', test_kl_loss, on_epoch=True, prog_bar=True, logger=True,
        #              batch_size=self.hparams.train_batch, sync_dist=True)
            
        #     target = torch.reshape(target, (-1, *target.shape[-3:]))
        #     filepath = np.transpose(filepath).flatten()
        #     self.all_filepaths.extend(filepath)
        #     for idx in range(data.shape[0]):
        #         comparison = torch.cat([
        #             target[idx, :, :, :128].unsqueeze(0),
        #             target[idx, :, :, 128:].unsqueeze(0),
        #             output[idx, :, :, :128].unsqueeze(0),
        #             output[idx, :, :, 128:].unsqueeze(0),
        #             next_output[idx, :, :, :128].unsqueeze(0),
        #             next_output[idx, :, :, 128:].unsqueeze(0)
        #         ])
        #         save_image(comparison.cpu(), os.path.join(self.pred_log_dir, filepath[idx]), nrow=1)
        #         latent_tmp = latent[idx].view(1, -1)[0].cpu().detach().numpy()
        #         self.all_latents.append(latent_tmp)
        #         mu_tmp = mu[idx].view(1, -1)[0].cpu().detach().numpy()
        #         self.all_mus.append(mu_tmp)
        #         logvar_tmp = logvar[idx].view(1, -1)[0].cpu().detach().numpy()
        #         self.all_logvars.append(logvar_tmp)
        #         next_latent_tmp = next_latent[idx].view(1, -1)[0].cpu().detach().numpy()
        #         self.all_next_latents.append(next_latent_tmp)

        return test_loss

    def test_save(self):
        if self.hparams.model_name in ['encoder-decoder-64', 'vit-tide']:
            np.save(os.path.join(self.var_log_dir, 'ids.npy'), self.all_filepaths)
            np.save(os.path.join(self.var_log_dir, 'latent.npy'), self.all_latents)
            np.save(os.path.join(self.var_log_dir, 'mu.npy'), self.all_mus)
            np.save(os.path.join(self.var_log_dir, 'logvar.npy'), self.all_logvars)
            np.save(os.path.join(self.var_log_dir, 'next_latent.npy'), self.all_next_latents)
        elif self.hparams.model_name in 'refine-64':
            np.save(os.path.join(self.var_log_dir, 'ids.npy'), self.all_filepaths)
            np.save(os.path.join(self.var_log_dir, 'latent.npy'), self.all_latents)
            np.save(os.path.join(self.var_log_dir, 'refine_latent.npy'), self.all_refine_latents)
            np.save(os.path.join(self.var_log_dir, 'next_refine_latent.npy'), self.all_next_refine_latents)
            np.save(os.path.join(self.var_log_dir, 'reconstructed_latent.npy'), self.all_reconstructed_latents)
        #     # Define the output video file path within the prediction directory.
        # video_output_path = os.path.join(self.pred_log_dir, "predictions_video.mp4")
    
        # # Get a sorted list of all .png files in the prediction folder.
        # img_files = sorted([os.path.join(self.pred_log_dir, f) for f in os.listdir(self.pred_log_dir)
        #                 if f.endswith('.png')])
    
        # if len(img_files) == 0:
        #     print("No prediction images found in", self.pred_log_dir, "to compile into a video.")
        #     return

        # # Set frames per second as desired (e.g., 10 fps)
        # fps = 10
        # writer = imageio.get_writer(video_output_path, fps=fps)
    
        # for f in img_files:
        #     img = imageio.imread(f)
        #     writer.append_data(img)
    
        # writer.close()
        # print("Saved video to:", video_output_path)
        video_output_path = os.path.join(self.pred_log_dir, "predictions_video.mp4")
        generate_video_from_predictions(self.pred_log_dir, video_output_path, fps=10)
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
    
    def paths_to_tuple(self, paths):
        new_paths = []
        for i in range(len(paths)):
            tmp = paths[i].split('.')[0].split('_')
            new_paths.append((int(tmp[0]), int(tmp[1])))
        return new_paths

    def setup(self, stage=None):

        if stage == 'fit':
            if self.hparams.model_name in ['encoder-decoder-64', 'vit-tide']:
                self.train_dataset = NeuralPhysDataset(
                    data_filepath=self.hparams.data_filepath,
                    num_frames=self.hparams.num_frames,
                    flag='train',
                    seed=self.hparams.seed,
                    object_name=self.hparams.dataset,
                )
                self.val_dataset = NeuralPhysDataset(
                    data_filepath=self.hparams.data_filepath,
                    num_frames=self.hparams.num_frames,
                    flag='val',
                    seed=self.hparams.seed,
                    object_name=self.hparams.dataset,
                )
            if self.hparams.model_name == 'refine-64':
                high_dim_var_log_dir = self.var_log_dir.replace('refine', 'encoder-decoder')
                train_mu = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_train', 'mu.npy')))
                train_logvar = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_train', 'logvar.npy')))
                train_mu = torch.unsqueeze(train_mu, dim=2)
                train_logvar = torch.unsqueeze(train_logvar, dim=2)
                train_data = torch.cat([train_mu, train_logvar], axis=2)
                train_target = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_train', 'mu.npy')))

                val_mu = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_val', 'mu.npy')))
                val_logvar = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_val', 'logvar.npy')))
                val_target = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_val', 'mu.npy')))
                val_mu = torch.unsqueeze(val_mu, dim=2)
                val_logvar = torch.unsqueeze(val_logvar, dim=2)
                val_data = torch.cat([val_mu, val_logvar], axis=2)
                val_target = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_val', 'mu.npy')))

                train_filepaths = list(np.load(os.path.join(high_dim_var_log_dir+'_train', 'ids.npy')))
                val_filepaths = list(np.load(os.path.join(high_dim_var_log_dir+'_val', 'ids.npy')))
                # convert the file strings into tuple so that we can use TensorDataset to load everything together
                train_filepaths = torch.Tensor(self.paths_to_tuple(train_filepaths))
                val_filepaths = torch.Tensor(self.paths_to_tuple(val_filepaths))
                self.train_dataset = NeuralPhysRefineDataset(
                    data_filepath=self.hparams.data_filepath,
                    object_name=self.hparams.dataset,
                    data=train_data,
                    target=train_target,
                    filepaths=train_filepaths,
                    num_frames=self.hparams.num_frames
                )
                self.val_dataset = NeuralPhysRefineDataset(
                    data_filepath=self.hparams.data_filepath,
                    object_name=self.hparams.dataset,
                    data=val_data,
                    target=val_target,
                    filepaths=val_filepaths,
                    num_frames=self.hparams.num_frames
                )

        if stage == 'test':
            self.test_dataset = NeuralPhysDataset(
                data_filepath=self.hparams.data_filepath,
                num_frames=self.hparams.num_frames,
                flag='test',
                seed=self.hparams.seed,
                object_name=self.hparams.dataset
            )
            # initialize lists for saving variables and latents during testing
            self.all_filepaths = []
            self.all_latents = []
            self.all_mus = []
            self.all_logvars = []
            self.all_next_latents = []
            self.all_refine_latents = []
            self.all_next_refine_latents = []
            self.all_reconstructed_latents = []

    # def train_dataloader(self):
    #     train_loader = torch.utils.data.DataLoader(
    #         dataset=self.train_dataset,
    #         batch_size=self.hparams.train_batch,
    #         shuffle=True,
    #         **self.kwargs
    #     )
    #     return train_loader

    # def val_dataloader(self):
    #     val_loader = torch.utils.data.DataLoader(
    #         dataset=self.val_dataset,
    #         batch_size=self.hparams.val_batch,
    #         shuffle=False,
    #         **self.kwargs
    #     )
    #     return val_loader

    # def test_dataloader(self):
    #     test_loader = torch.utils.data.DataLoader(
    #         dataset=self.test_dataset,
    #         batch_size=self.hparams.test_batch,
    #         shuffle=False,
    #         **self.kwargs
    #     )
    #     return test_loader
    def train_dataloader(self):
        kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True} if self.hparams.if_cuda else {}
        train_dataset = NeuralPhysDataset(
            data_filepath=self.hparams.data_filepath,
            num_frames=self.hparams.num_frames,
            flag='train',
            seed=self.hparams.seed,
            object_name=self.hparams.dataset
        )
        return DataLoader(train_dataset, batch_size=self.hparams.train_batch, shuffle=True, **kwargs)

    def val_dataloader(self):
        kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True} if self.hparams.if_cuda else {}
        val_dataset = NeuralPhysDataset(
            data_filepath=self.hparams.data_filepath,
            num_frames=self.hparams.num_frames,
            flag='val',
            seed=self.hparams.seed,
            object_name=self.hparams.dataset
        )
        return DataLoader(val_dataset, batch_size=self.hparams.val_batch, shuffle=False, **kwargs)

    def test_dataloader(self):
        kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True} if self.hparams.if_cuda else {}
        test_dataset = NeuralPhysDataset(
            data_filepath=self.hparams.data_filepath,
            num_frames=self.hparams.num_frames,
            flag='test',
            seed=self.hparams.seed,
            object_name=self.hparams.dataset
        )
        return DataLoader(test_dataset, batch_size=self.hparams.test_batch, shuffle=False, **kwargs)