import torch
import random


class LatentSampler():
    def __init__(self, generator, config):
        self.config = config
        self.generator = generator

    @torch.no_grad()
    def sample_global_latent(self, batch_size, device, requires_grad=False, mixing=True):
        global_latent_dim = self.config.train_params.global_latent_dim
        is_mixing = random.random() < self.config.train_params.mixing if mixing else False

        latent_1 = torch.randn(batch_size, global_latent_dim, device=device)
        latent_2 = torch.randn(batch_size, global_latent_dim, device=device)
        latent = torch.stack([
            latent_1,
            latent_2 if is_mixing else latent_1,
        ], 1) # shape: (B, 2, D) # batch-first for dataparallel

        latent.requires_grad = requires_grad
        return latent

    def sample_local_latent(self, batch_size, device, requires_grad=False,
                            spatial_size_enlarge=1, specific_shape=None, exclude_padding=False):

        local_latent_dim = self.config.train_params.local_latent_dim   

        if specific_shape is not None:
            spatial_shape = specific_shape
        elif spatial_size_enlarge != 1:
            if hasattr(self.config.train_params, "styleGAN2_baseline") and self.config.train_params.styleGAN2_baseline:
                size = self.config.train_params.ts_input_size * spatial_size_enlarge
                spatial_shape = (size, size)
            else:
                base = self.config.train_params.ts_input_size // 2
                size = (int(round(base * spatial_size_enlarge)) * 2) + 1
                spatial_shape = (size, size)
        else:
            size = self.config.train_params.ts_input_size
            spatial_shape = (size, size)
        
        if self.config.train_params.use_ss and self.config.train_params.ss_unfold_radius > 0:
            if self.config.train_params.ss_n_layers > 0:
                ss_unfold_size = self.config.train_params.ss_n_layers * self.config.train_params.ss_unfold_radius
            else:
                ss_unfold_size = 0
            if exclude_padding:
                spatial_shape_ext = spatial_shape
            else:
                spatial_shape_ext = [
                    spatial_shape[0] + 2 * ss_unfold_size,
                    spatial_shape[1] + 2 * ss_unfold_size]
            z_local = torch.randn(batch_size, local_latent_dim, spatial_shape_ext[0], spatial_shape_ext[1], device=device)
        else:
            z_local = torch.randn(batch_size, local_latent_dim, spatial_shape[0], spatial_shape[1], device=device)

        z_local.requires_grad = requires_grad
        return z_local

