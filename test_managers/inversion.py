import os
import math
import numpy as np
import pickle as pkl
from numpy.linalg import inv
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision.utils import make_grid

from libs import lpips
from utils import *
from models.diff_augment_dual import DiffAugmentDual
from models.losses import l2_loss, noise_regularize, calc_path_lengths, g_path_regularize
from dataset import MultiResolutionDataset, DictTensor
from test_managers.base_test_manager import BaseTestManager
from test_managers.global_config import test_meta_extra_pad


pix_dist_metric = l2_loss


def extract_training_params(g_ema, inject_randomness=False):
    params = []
    exclude_keys = [
        "modulation", # Causes instability
        "activate", # FusedLeakyReLU has a bias parameter
        "noise", # The strength of noise injection
    ]
    for name,param in g_ema.named_parameters():
        is_exclude = False
        for ex_k in exclude_keys:
            if ex_k in name:
                is_exclude = True
                break
        if not is_exclude:
            params.append(param)

            if inject_randomness:
                mean_, var_ = param.data.mean().item(), param.data.var().item()
                randomness = torch.randn(*param.data.shape) * var_ * 0.001 + mean_
                param.data.add_(randomness.to(param.data.device))
    return params 


class InversionManager(BaseTestManager):

    def task_specific_init(self):

        self.target_height = self.config.task.height
        self.target_width = self.config.task.width

        # [NOTE] Not using sequential generation here, but the logic of computing meta-shape is the same.
        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        # Get all the starting points of the patches to generate
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + 1
        self.num_steps_w = \
            math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + 1
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.pixelspace_step_size * (self.num_steps_w-1) + self.outfeat_sizes_list[-1]
 
        self.ss_unfold_size = self.config.train_params.ss_n_layers * self.config.train_params.ss_unfold_radius

        """
        Dataset
        """
        config_clone = deepcopy(self.config)
        if hasattr(self.config.task, "img_dir"):
            dataset = MultiResolutionDataset(
                img_dir=self.config.task.img_dir,
                config=config_clone,
                is_training=False,
                simple_return_full=True)
        elif hasattr(self.config.task, "data_split"):
            dataset = MultiResolutionDataset(
                split=self.config.task.data_split,
                config=config_clone,
                is_training=False,
                simple_return_full=True)
        else:
            raise ValueError("Either `img_dir` or `data_split` should be specified in the test_config!")
        self.dataloader = iter(data.DataLoader(
            dataset,
            batch_size=self.config.task.batch_size,
            sampler=data_sampler(dataset, shuffle=False, init_index=self.config.task.init_index),
            drop_last=False,
        ))
        self.dataset_size = len(dataset)

        """
        Get latent statistics
        """
        n_random_samples = 10000
        global_latents = self.latent_sampler.sample_global_latent(
            n_random_samples, 
            device=self.device) # (B, 2, C)
        ts_styles = self.g_ema(
            call_internal_method="get_style", 
            internal_method_kwargs={"global_latent": global_latents[:, 0]})
        ts_styles_gau = self._gaussianize_latents(ts_styles)

        self.ts_styles_gau_mean = ts_styles_gau.mean(0) # (C, )
        self.ts_styles_gau_std = \
            ((ts_styles_gau - self.ts_styles_gau_mean).pow(2).sum(0) / n_random_samples) ** 0.5 # float

        print(" [*] TS Style (Gaussianized) mean:", self.ts_styles_gau_mean.mean())
        print(" [*] TS Style (Gaussianized) std:", self.ts_styles_gau_std.mean())

        """
        Statistics for the prior loss of Gaussianized latent space
        """
        prior_gau_invcov = inv(np.cov(ts_styles_gau.cpu().numpy(), rowvar=False))
        self.ts_styles_gau_invcov = torch.from_numpy(prior_gau_invcov).float().to(self.device) # (C, C)

        """
        Loss
        """
        n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        self.percept = lpips.PerceptualLoss(
            model="net-lin", 
            net="vgg", 
            gpu_ids=np.arange(n_gpus).tolist(), # default supports dataparallel
        )

        """
        Logging
        """
        if self.config.task.write_tb_logs:
            self.log_dir = os.path.join(
                "logs-inv", 
                self.config.var.exp_name, 
                self.config.task.config_name)
            if self.config.task.exp_suffix is not None:
                self.log_dir += ("_" + self.config.task.exp_suffix)
            else: # auto suffix by loss weights
                auto_suffix = "_".join([
                    f"LR{self.config.task.lr}",
                    f"PixL{self.config.task.losses_weights.pixel_dist}",
                    f"PerL{self.config.task.losses_weights.lpips_dist}",
                    f"NReg{self.config.task.losses_weights.noise_reg}",
                    f"SSGPrior{self.config.task.losses_weights.ss_global_prior_loss}",
                    f"SSLPrior{self.config.task.losses_weights.ss_local_prior_loss}",
                    f"SSGStd{self.config.task.losses_weights.ss_global_std_loss}",
                    f"SSLStd{self.config.task.losses_weights.ss_local_std_loss}",
                    f"TSPrior{self.config.task.losses_weights.ts_prior_loss}",
                ])
                if self.config.task.diff_aug:
                    auto_suffix += "_DA"
                if self.config.task.noise_renorm == False:
                    auto_suffix += "_NoNoiseNorm"
                if hasattr(self.config.task, "local_latent_renorm") and self.config.task.local_latent_renorm:
                    auto_suffix += "_SSLNorm"
                if self.config.task.learned_coords:
                    auto_suffix += "_LearnedCoords"
                self.log_dir += ("_" + auto_suffix)

            if (not os.path.exists(self.log_dir)):
                os.makedirs(self.log_dir)

            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None


    @torch.enable_grad()
    def run_next(self, save=False, **kwargs):

        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if v is not None:
                    print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))

        """
        Get target image
        """
        real_data = DictTensor(next(self.dataloader)).to(self.device)
        gt_imgs = real_data["full"]
        cur_batch_size, C, H, W = gt_imgs.shape # Last may have smaller batch size
        print(" [*] Real image target shape: ({}, {})".format(H, W))

        if hasattr(self.config.task, "inv_region"):
            assert not hasattr(self.config.task, "inv_mode")
            assert hasattr(self.config.task, "eval_region")
            assert hasattr(self.config.task, "center_drift")
            xst, yst, xed, yed = self.config.task.inv_region
            target_imgs = gt_imgs[:, :, xst:xed, yst:yed]
            center_drift = self.config.task.center_drift
        elif self.config.task.inv_mode == "l2r": # invert left-half, outpaint right-half
            target_imgs = gt_imgs[:, :, :, :W//2]
            center_drift = (0, -(W//4))
        elif self.config.task.inv_mode == "r2l": # invert right-half, outpaint left-half
            target_imgs = gt_imgs[:, :, :, W//2:]
            center_drift = (0, W//4)
        elif self.config.task.inv_mode == "u2b": # invert upper-half, outpaint bottom-half
            target_imgs = gt_imgs[:, :, :H//2, :]
            center_drift = (-(H//4), 0)
        elif self.config.task.inv_mode == "b2u": # invert bottom-half, outpaint upper-half
            target_imgs = gt_imgs[:, :, H//2:, :]
            center_drift = (H//4, 0)
        elif self.config.task.inv_mode == "all": # invert the whole image
            target_imgs = gt_imgs
            center_drift = (0, 0)
        else:
            raise NotImplementedError("Unknown inv_mode {}".format(self.config.task.inv_mode))

        """
        Create SS (structure_synthesizer) inputs
        """
        full_structure_latent_shape = (
            # Does not account GNN padding here, it is handled within the latent_sampler
            int(self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)),
            int(self.g_ema_module.calc_in_spatial_size(self.meta_width, include_ss=False)),
        )
        ss_global_latents = self.latent_sampler.sample_global_latent(
            cur_batch_size, 
            mixing=False, 
            device=self.device,
            requires_grad=True)
        ss_local_latents = self.latent_sampler.sample_local_latent(
            cur_batch_size, 
            device=self.device,
            requires_grad=True,
            specific_shape=full_structure_latent_shape)
        if hasattr(self.config.task, "future_placement"):
            coords = self.get_future_coords(ss_local_latents)
        else:
            coords = self.coord_handler.sample_coord_grid(
                ss_local_latents, 
                is_training=False) # is_training=False creates zero-centered coords without using cache.
        if self.config.task.learned_coords:
            coords.requires_grad = True

        """
        Create TS (texture_synthesizer) inputs
        """
        num_latents = self.g_ema_module.texture_synthesizer.n_latent
        ts_styles_init = self._de_gaussianize_latents(self.ts_styles_gau_mean)
        ts_styles = ts_styles_init.detach().clone().repeat(cur_batch_size, num_latents, 1)
        ts_styles.requires_grad = True

        noise_size_list_h = self.g_ema_module.calc_out_spatial_size(
            full_structure_latent_shape[0], include_ss=False, return_list=True)
        noise_size_list_w = self.g_ema_module.calc_out_spatial_size(
            full_structure_latent_shape[1], include_ss=False, return_list=True)
        ts_noises = [
            torch.randn(cur_batch_size, 1, int(h), int(w), device=self.device, requires_grad=True)
                for (h,w) in zip(noise_size_list_h, noise_size_list_w)]

        """
        Create optimizer
        """
        ss_lr_mult = 0.1
        ts_lr_mult = 1
        ss_opt_params = [ss_global_latents, ss_local_latents]
        ts_opt_params = [ts_styles, *ts_noises]
        if self.config.task.learned_coords:
            ss_opt_params += [coords]
        ss_optimizer = torch.optim.Adam(ss_opt_params, lr=self.config.task.lr*ss_lr_mult)
        ts_optimizer = torch.optim.Adam(ts_opt_params, lr=self.config.task.lr*ts_lr_mult)

        if hasattr(self.config.task, "tqdm") and self.config.task.tqdm:
            pbar = tqdm(range(self.config.task.n_steps))
        else:
            pbar = range(self.config.task.n_steps)

        # [TODO] write some losses in cmdline might be a good idea
        for iter_ in pbar:

            # LR ramp-up / ramp-down
            progress_ratio = iter_ / self.config.task.n_steps
            ts_optimizer.param_groups[0]["lr"] = \
                self._get_cur_lr(progress_ratio, self.config.task.lr*ts_lr_mult)
            for i in range(len(ss_optimizer.param_groups)):
                ss_optimizer.param_groups[i]["lr"] = \
                    self._get_cur_lr(progress_ratio, self.config.task.lr*ss_lr_mult)

            # Calibrate latents with randomized perturbation for early-stage exploration
            ramp_factor = max(0, 1 - progress_ratio / self.config.task.rand_perturb_ramp) ** 2
            ts_styles_rand_strength = self.ts_styles_gau_std.mean() * self.config.task.rand_styles_perturb * ramp_factor
            ts_styles_rand = self._rand_perturb_latents(ts_styles, progress_ratio, ts_styles_rand_strength)

            # Randomly perturb ss_local, see if reduces overfitting for outward pixels
            if hasattr(self.config.task, "rand_ss_local_perturb") and self.config.task.rand_ss_local_perturb > 0:
                ramp_factor = max(0, 1 - progress_ratio / self.config.task.rand_ss_local_perturb_ramp) ** 2
                ss_local_rand_strength = self.config.task.rand_ss_local_perturb * ramp_factor
                ss_local_latents_rnd = self._rand_perturb_latents(ss_local_latents, progress_ratio, ss_local_rand_strength)
            else:
                ss_local_latents_rnd = ss_local_latents
                ss_local_rand_strength = 0

            if hasattr(self.config.task, "rand_ss_global_perturb") and self.config.task.rand_ss_global_perturb > 0:
                ramp_factor = max(0, 1 - progress_ratio / self.config.task.rand_ss_global_perturb_ramp) ** 2
                ss_global_rand_strength = self.config.task.rand_ss_global_perturb * ramp_factor
                ss_global_latents_rnd = self._rand_perturb_latents(ss_global_latents, progress_ratio, ss_global_rand_strength)
            else:
                ss_global_latents_rnd = ss_global_latents
                ss_global_rand_strength = 0

            # Two-stage inference here, since SS and TS may use different global latents
            inter_output = self.g_ema(
                global_latent=ss_global_latents_rnd,
                local_latent=ss_local_latents_rnd,
                override_coords=coords,
                early_return_structure_latent=True,
                disable_dual_latents=True)
            output = self.g_ema(
                styles=ts_styles_rand, 
                structure_latent=inter_output["structure_latent"],
                override_coords=coords,
                noises=ts_noises,
                disable_dual_latents=True)

            inv_raw = self.output_size_coorection(output["gen"])
            inv_results = self._crop_aligned_region(inv_raw, target_imgs, center_drift)

            # Project latents from Gaussianized latent space to skewed StyleGAN2 latent space
            ts_styles_gau = self._gaussianize_latents(ts_styles)

            losses = self.compute_losses(
                ss_local_latents_rnd, ss_global_latents_rnd, coords, 
                ts_styles, ts_styles_gau, ts_noises, 
                target_imgs, inv_results, inv_raw)

            ts_optimizer.zero_grad()
            ss_optimizer.zero_grad()
            total_loss = self._loss_agg(losses, progress_ratio)
            total_loss.backward()
            for k,loss in losses.items():
                if torch.isnan(loss.mean()).item() or torch.isinf(loss.mean()).item():
                    print(" [!] Inf or NaN at {}".format(k))
                    import pdb; pdb.set_trace()
            if torch.isnan(ss_global_latents.grad).any() or torch.isinf(ss_global_latents.grad).any():
                import pdb; pdb.set_trace()
            ts_optimizer.step()
            ss_optimizer.step()

            # Post-process and calibrate some variables inplace
            self.calibrate_variables(ss_local_latents, coords, ts_noises)

            # Eval and logging
            with torch.no_grad():                    
                inv_raw_anno = self._annotate_aligned_region(
                    inv_raw, target_imgs, center_drift=center_drift)
                inv_raw_comp = self._direct_compose(
                    inv_raw, target_imgs, center_drift=center_drift)
            if self.config.task.eval and (iter_ % self.config.task.log_value_steps == 0 or iter_ == self.config.task.n_steps-1):
                with torch.no_grad():                    
                    if hasattr(self.config.task, "eval_extra") and self.config.task.eval_extra == False:
                        eval_scores = {}
                    else:
                        inv_results_toward_gt = self._crop_aligned_region(inv_raw, gt_imgs)
                        eval_scores = self.eval(inv_results_toward_gt, gt_imgs)
                    eval_scores_reduced = {k: v.mean() for k,v in eval_scores.items()}
                    logging_distrs = {
                        "x_coords": coords[:, 0].mean([1,2]),
                        "y_coords": coords[:, 1].mean([1,2]),
                        "noises": torch.cat([noise.reshape(-1) for noise in ts_noises]),
                        "ss_local_latents": ss_local_latents,
                        "ss_global_latents": ss_global_latents,
                        "ts_styles": ts_styles,
                        "ts_styles_gau": ts_styles_gau,
                    }
                    logging_distrs = {
                        k: self._general_toarray(v)
                            for k,v in logging_distrs.items() if (v is not None)}
                    other_scalars = {
                        "lr": ts_optimizer.param_groups[0]["lr"],
                        "ts_styles_rand_strength": ts_styles_rand_strength,
                        "ss_local_rand_strength": ss_local_rand_strength,
                    }

                    trait_imgs = self._create_trait_variations(
                        ss_global_latents, ss_local_latents, coords, ts_styles, ts_noises, 
                        inv_raw, full_structure_latent_shape, target_imgs, center_drift)

                self.maybe_logging(
                    losses, eval_scores_reduced, logging_distrs, other_scalars,
                    target_imgs, inv_results, inv_raw, inv_raw_anno, inv_raw_comp, gt_imgs, trait_imgs, iter_)
        
        if save:
            print(" [*] Saving results...")
            self.save_results({
                "real_gt": self._pt_to_img(gt_imgs),
                "inv_cmp": self._pt_to_img(torch.cat([inv_results, target_imgs], 3)),
                "inv_raw": self._pt_to_img(inv_raw),
                "inv_raw_anno": self._pt_to_img(inv_raw_anno),
                "inv_raw_comp": self._pt_to_img(inv_raw_comp),
            })

            # Save eval scores, might be helpful for paper figures picking
            final_latents = {
                "ss_local_latents": ss_local_latents,
                "ss_global_latents": ss_global_latents,
                "ts_styles": ts_styles,
                "ts_styles_gau": ts_styles_gau,
                "ts_noises": ts_noises,
                "coords": coords,
            }
            if (not self.config.task.eval):
                eval_scores = None
            self.save_stats(eval_scores, final_latents)

        self.cur_global_id += cur_batch_size


    def _general_toarray(self, v):
        if isinstance(v, list):
            return [vv.detach().cpu().numpy() for vv in v]
        else:
            return v.detach().cpu().numpy()


    def eval(self, inv_results, gt_imgs):

        h_center = gt_imgs.shape[2]//2
        w_center = gt_imgs.shape[3]//2
        if hasattr(self.config.task, "inv_region"):
            inv_xst, inv_yst, inv_xed, inv_yed = self.config.task.inv_region
            eva_xst, eva_yst, eva_xed, eva_yed = self.config.task.eval_region
            reconstr_real = gt_imgs[:, :, inv_xst:inv_xed, inv_yst:inv_yed]
            reconstr_fake = inv_results[:, :, inv_xst:inv_xed, inv_yst:inv_yed]
            outpaint_real = gt_imgs[:, :, eva_xst:eva_xed, eva_yst:eva_yed]
            outpaint_fake = inv_results[:, :, eva_xst:eva_xed, eva_yst:eva_yed]
        elif self.config.task.inv_mode == "l2r": # invert left-half, outpaint right-half
            reconstr_real = gt_imgs[:, :, :, :h_center]
            reconstr_fake = inv_results[:, :, :, :h_center]
            outpaint_real = gt_imgs[:, :, :, h_center:]
            outpaint_fake = inv_results[:, :, :, h_center:]
        elif self.config.task.inv_mode == "r2l": # invert right-half, outpaint left-half
            reconstr_real = gt_imgs[:, :, :, h_center:]
            reconstr_fake = inv_results[:, :, :, h_center:]
            outpaint_real = gt_imgs[:, :, :, :h_center]
            outpaint_fake = inv_results[:, :, :, :h_center]
        elif self.config.task.inv_mode == "u2b": # invert upper-half, outpaint bottom-half
            reconstr_real = gt_imgs[:, :, :w_center, :]
            reconstr_fake = inv_results[:, :w_center, :, :]
            outpaint_real = gt_imgs[:, :, w_center:, :]
            outpaint_fake = inv_results[:, :, w_center:, :]
        elif self.config.task.inv_mode == "b2u": # invert bottom-half, outpaint upper-half
            reconstr_real = gt_imgs[:, :, w_center:, :]
            reconstr_fake = inv_results[:, w_center:, :, :]
            outpaint_real = gt_imgs[:, :, :w_center, :]
            outpaint_fake = inv_results[:, :, :w_center, :]
        elif self.config.task.inv_mode == "all": # invert the whole image
            reconstr_real = gt_imgs
            reconstr_fake = inv_results
            outpaint_real = None
            outpaint_fake = None
        else:
            raise NotImplementedError("Unknown inv_mode {}".format(self.config.task.inv_mode))

        eval_scores = {
            "reconstr_pixel": 
                pix_dist_metric(reconstr_real, reconstr_fake),
            "reconstr_lpips": 
                self.percept(reconstr_real, reconstr_fake).squeeze(),
            "outpaint_pixel": 
                pix_dist_metric(outpaint_real, outpaint_fake) if outpaint_real is not None else None,
            "outpaint_lpips": 
                self.percept(outpaint_real, outpaint_fake).squeeze() if outpaint_real is not None else None,
        }

        return eval_scores

    """
    Losses and regularizers
    """
    def compute_losses(self, ss_local_latents, ss_global_latents, coords, 
                             ts_styles, ts_styles_gau, ts_noises, 
                             target_imgs, inv_results, inv_raw):

        if self.config.task.diff_aug:
            inv_results, target_imgs = \
                DiffAugmentDual(inv_results, target_imgs, policy="color,translation,cutout")

        bs, ssl_c, ssl_h, ssl_w = ss_local_latents.shape
        losses = {
            "pixel_dist": pix_dist_metric(inv_results, target_imgs, reduce_all=True),
            "lpips_dist": self.percept(inv_results, target_imgs).mean(),
            # [TODO] Actually, we probably should reg the used region only?
            "noise_reg": noise_regularize(ts_noises),

            "ss_global_prior_loss": self.unit_gaussian_prior_loss(ss_global_latents[:, 0], mode="mean"),
            "ss_local_prior_loss": self.unit_gaussian_prior_loss(ss_local_latents, mode="l2"),

            "ss_global_std_loss": self.unit_gaussian_std_loss(ss_global_latents[:, 0]),
            "ss_local_std_loss": self.unit_gaussian_std_loss(ss_local_latents),
        }

        losses["ts_prior_loss"] = self.empirical_gaussian_prior_loss(
            ts_styles_gau,
            self.ts_styles_gau_mean, 
            self.ts_styles_gau_std,
            self.ts_styles_gau_invcov)

        # # Unused
        # losses["ts_mean_loss"] = self.empirical_gaussian_mean_loss(
        #     ts_styles_gau,
        #     self.ts_styles_gau_mean, 
        #     self.ts_styles_gau_std,
        #     self.ts_styles_gau_invcov)

        return losses

    def calibrate_variables(self, local_latent, coords, ts_noises):
        if self.config.task.noise_renorm:
            self._noise_renormalize(ts_noises)
        # # Unused
        # if hasattr(self.config.task, "local_latent_renorm") and self.config.task.local_latent_renorm:
        #     self._local_latent_renormalize(local_latent)

        # [TODO] probably should compute the updated part only
        if self.config.task.learned_coords:
            dirty_mean = coords.mean([2,3]) 
            self.coord_handler.update_coords_by_mean(coords, dirty_mean)

    def _noise_renormalize(self, noises, eps=1e-4):
        for noise in noises:
            mean = noise.mean()
            std = noise.std()
            if std.item() < 1e-4 or std.item() > 1e6:
                print("NaN", noise.mean(), noise.std())
                import pdb; pdb.set_trace()
            std = eps if std.item() < eps else std
            noise.data.add_(-mean).div_(std)
        
    def _local_latent_renormalize(self, local_latent, eps=1e-4):
        mean = local_latent.mean()
        std = local_latent.std()
        if std.item() < 1e-4 or std.item() > 1e6:
            print("NaN", noise.mean(), noise.std())
            import pdb; pdb.set_trace()
        std = eps if std.item() < eps else std
        # mean = local_latent.mean([2,3], keepdim=True)
        # std = local_latent.std([2,3], keepdim=True)
        # if (std < eps).any():
        #     std[std < eps] += eps
        local_latent.data.add_(-mean).div_(std)

    def _rand_perturb_latents(self, latents, progress_ratio, strength):
        new_latents = latents + torch.randn_like(latents) * strength
        return new_latents

    def _get_cur_lr(self, progress_ratio, initial_lr, rampdown=0.25, rampup=0.05):
        lr_ramp = min(1, (1 - progress_ratio) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, progress_ratio / rampup)
        return initial_lr * lr_ramp

    def _crop_aligned_region(self, source, reference, center_drift=(0, 0)):
        _, _, h_size, w_size = reference.shape
        h_crop_st = (source.shape[-2] - reference.shape[-2]) // 2 + center_drift[0]
        w_crop_st = (source.shape[-1] - reference.shape[-1]) // 2 + center_drift[1]
        cropped = source[:, :, h_crop_st:h_crop_st+h_size, w_crop_st:w_crop_st+w_size]
        return cropped

    def _annotate_aligned_region(self, source, reference, center_drift=(0, 0), color="r"):
        assert color.lower() in {"r", "g", "b"}
        if color.lower() == "r":
            anno_cdim = 0
        elif color.lower() == "g":
            anno_cdim = 1
        elif color.lower() == "b":
            anno_cdim = 2

        _, _, h_size, w_size = reference.shape
        h_st = (source.shape[-2] - reference.shape[-2]) // 2 + center_drift[0] - 1
        w_st = (source.shape[-1] - reference.shape[-1]) // 2 + center_drift[1] - 1
        h_ed = h_st + h_size + 2
        w_ed = w_st + w_size + 2

        new_source = source.detach().clone()
        if h_st >= 0:
            new_source[:, :, h_st, w_st:w_ed] = 0 # top
        if h_ed < new_source.shape[2]:
            new_source[:, :, h_ed, w_st:w_ed] = 0 # bottom
        if w_st >= 0:
            new_source[:, :, h_st:h_ed, w_st] = 0 # left
        if w_ed < new_source.shape[3]:
            new_source[:, :, h_st:h_ed, w_ed] = 0 # right

        if h_st >= 0:
            new_source[:, anno_cdim, h_st, w_st:w_ed] = 1 # top
        if h_ed < new_source.shape[2]:
            new_source[:, anno_cdim, h_ed, w_st:w_ed] = 1 # bottom
        if w_st >= 0:
            new_source[:, anno_cdim, h_st:h_ed, w_st] = 1 # left
        if w_ed < new_source.shape[3]:
            new_source[:, anno_cdim, h_st:h_ed, w_ed] = 1 # right
        return new_source

    def _direct_compose(self, source, reference, center_drift=(0, 0)):
        _, _, h_size, w_size = reference.shape
        h_st = (source.shape[-2] - reference.shape[-2]) // 2 + center_drift[0]
        w_st = (source.shape[-1] - reference.shape[-1]) // 2 + center_drift[1]
        new_source = source.detach().clone()
        new_source[:, :, h_st:h_st+h_size, w_st:w_st+w_size] = reference
        return new_source

    def _weight_regularization(self):
        if "g_ema" in self.weight_reg_state_dict: # if dataparallel
            state_dict = self.weight_reg_state_dict["g_ema"]
        else:
            state_dict = self.weight_reg_state_dict
        total_norm = torch.tensor(0).float().to(self.device)
        count = 0
        new_params = dict(self.g_ema.named_parameters())
        for k,ori_param in state_dict.items():
            if k in new_params:
                new_param = new_params[k]
                diff_norm = (new_param - ori_param).norm(2)
                if diff_norm.item() > 1e-6:
                    total_norm += diff_norm
                    count += 1
        if count > 0:
            return total_norm / count
        else:
            return total_norm
    
    """
    Gaussianized latent space
    """
    def _gaussianize_latents(self, latents):
        if isinstance(latents, list):
            return [F.leaky_relu(v, negative_slope=5)*(1/2**0.5) for v in latents]
        else:
            return F.leaky_relu(latents, negative_slope=5)*(1/2**0.5)

    def _de_gaussianize_latents(self, latents):
        if isinstance(latents, list):
            return [F.leaky_relu(v / (1/2**0.5), negative_slope=0.2) for v in latents]
        else:
            return F.leaky_relu(latents / (1/2**0.5), negative_slope=0.2)

    def unit_gaussian_std_loss(self, latents):
        loss = torch.sqrt(((latents.std()-1)**2).mean())
        if loss.item() < 1e-6:
            return torch.zeros_like(loss)
        else:
            return loss

    def unit_gaussian_prior_loss(self, latents, variance_lb=0, mode="l2"):
        if variance_lb == 0:
            if mode=="l2":
                loss = torch.sqrt((latents**2).mean())
            elif mode=="mean":
                loss = latents.mean().abs()
            else:
                raise NotImplementedError(mode)
        else:
            latents = F.relu(latents.abs()-variance_lb)
            loss = torch.sqrt((latents**2).mean())
        return loss

    def empirical_gaussian_prior_loss(self, latents, empirical_mean, empirical_std, empirical_invcov):
        """
        [Let L be the number of latents in texture synthesizer]
        latents: shape (B, L, C)
        empirical_mean: shape (C) or (L, C)
        empirical_invcov: shape (C, C) or (L, C, C)

        is_unit_gaussian: whether directly assume all dimensions are N(0, 1)
        """
        prior_loss = 0
        if isinstance(latents, list):
            n_layers = len(latents)
        else:
            n_layers = latents.shape[1]
        for j in range(n_layers):
            if isinstance(latents, list):
                cur_style = latents[j]
                cur_style = cur_style.mean([2, 3]) # Collapse (N, C, H, W) => (N, C)
            else:
                cur_style = latents[:, j, :]
            if empirical_mean.ndim == 1:
                cur_empirical_mean = empirical_mean.unsqueeze(0) # (1xC)
                cur_empirical_invcov = empirical_invcov # (CxC)
            elif empirical_mean.ndim == 2: # each layer has different prior mean
                cur_empirical_mean = empirical_mean[j:j+1] # (1xC)
                cur_empirical_invcov = empirical_invcov[j, :, :] # (CxC)
            else:
                raise NotImplementedError()

            # The correct implementation, but the `invcov` has a numerical issue (+- 1e8~1e6)
            mu_dist = (cur_style - cur_empirical_mean) # (BxC) - (1xC) => (BxC)
            mu_dist = mu_dist.unsqueeze(2) # (BxCx1)
            # (Bx1xC) @ (CxC) @ (BxCx1)
            # => (Bx1xC) @ (BxCx1)
            # => (Bx1x1)
            prior_loss += torch.sqrt((mu_dist.transpose(1, 2) @ cur_empirical_invcov @ mu_dist).mean())
            
        return prior_loss / n_layers

    def empirical_gaussian_mean_loss(self, latents, empirical_mean, empirical_std, empirical_invcov):
        """
        [Let L be the number of latents in texture synthesizer]
        latents: shape (B, L, C)
        empirical_mean: shape (C) or (L, C)
        empirical_invcov: shape (C, C) or (L, C, C)

        is_unit_gaussian: whether directly assume all dimensions are N(0, 1)
        """
        prior_loss = 0
        if isinstance(latents, list):
            n_layers = len(latents)
        else:
            n_layers = latents.shape[1]
        for j in range(n_layers):
            if isinstance(latents, list):
                cur_style = latents[j]
                cur_style = cur_style.mean([2, 3]) # Collapse (N, C, H, W) => (N, C)
            else:
                cur_style = latents[:, j, :]
            if empirical_mean.ndim == 1:
                cur_empirical_mean = empirical_mean.unsqueeze(0) # (1xC)
                cur_empirical_invcov = empirical_invcov # (CxC)
            elif empirical_mean.ndim == 2: # each layer has different prior mean
                cur_empirical_mean = empirical_mean[j:j+1] # (1xC)
                cur_empirical_invcov = empirical_invcov[j, :, :] # (CxC)
            else:
                raise NotImplementedError()

            # The correct implementation, but the `invcov` has a numerical issue (+- 1e8~1e6)
            mu_dist = (cur_style - cur_empirical_mean) # (BxC) - (1xC) => (BxC)
            weight = self.ts_styles_gau_std / self.ts_styles_gau_std.mean()
            prior_loss += (weight * mu_dist.mean(0)).abs().mean()
            
        return prior_loss / n_layers

    """
    Logging and saving
    """
        
    def maybe_logging(self, losses, eval_scores_reduced, logging_distrs, other_scalars,
                      target_imgs, inv_results, inv_raw, inv_raw_anno, inv_raw_comp, gt_imgs, trait_imgs, iter_):
        if self.cur_global_id >= self.config.task.log_n_samples:
            return

        if self.writer is None:
            return

        if iter_ % self.config.task.log_value_steps == 0 or iter_ == self.config.task.n_steps-1:
            try:
                for k,v in losses.items():
                    self.writer.add_scalar("losses/"+k, v.item(), iter_)
                for k,v in eval_scores_reduced.items():
                    self.writer.add_scalar("eval/"+k, v.item(), iter_)
                for k,v in other_scalars.items():
                    self.writer.add_scalar("utils/"+k, v, iter_)
                for k,v in logging_distrs.items():
                    if isinstance(v, list):
                        v = np.concatenate([vv.reshape(-1) for vv in v], 0)
                        v = v[::100] # Too large, subsample
                    self.writer.add_histogram(k, v, iter_)
            except Exception as e:
                print(e)
                print(" [key] => {}".format(k))
                import pdb; pdb.set_trace()
                raise e

        if iter_ % self.config.task.log_img_steps == 0 or iter_ == self.config.task.n_steps-1:
            inv_raw = inv_raw.detach().cpu()
            inv_results = inv_results.detach().cpu()
            target_imgs = target_imgs.detach().cpu()

            img_t = torch.cat([inv_results, target_imgs], 3).clamp(-1, 1)
            img_t = make_grid(img_t, nrow=1, normalize=True, range=(-1, 1))
            self.writer.add_image("inv_cmp", img_t, iter_)

            img_t = make_grid(inv_raw, nrow=1, normalize=True, range=(-1, 1))
            self.writer.add_image("inv_raw", img_t, iter_)

            img_t = make_grid(inv_raw_anno, nrow=1, normalize=True, range=(-1, 1))
            self.writer.add_image("inv_raw_anno", img_t, iter_)

            img_t = make_grid(inv_raw_comp, nrow=1, normalize=True, range=(-1, 1))
            self.writer.add_image("inv_raw_comp", img_t, iter_)
            
            img_t = make_grid(gt_imgs, nrow=1, normalize=True, range=(-1, 1))
            self.writer.add_image("real_gt", img_t, iter_)

            # Already processed with make_grid
            if trait_imgs is not None:
                self.writer.add_image("trait_imgs", trait_imgs, iter_)

        if iter_ == self.config.task.n_steps-1:
            img_t = make_grid(inv_raw, nrow=1, normalize=True, range=(-1, 1))
            self.writer.add_image("final_inv_raw", img_t, self.cur_global_id)

    def save_results(self, results_dict):

        for result_name, imgs in results_dict.items():
            if imgs is None: continue

            subdir = os.path.join(self.save_root, "imgs", result_name)
            if (not os.path.exists(subdir)): os.makedirs(subdir)

            for i in range(imgs.shape[0]):
                sample_id = self.cur_global_id + i
                img_name = str(sample_id).zfill(6) + ".png"
                plt.imsave(os.path.join(subdir, img_name), imgs[i])

    def save_stats(self, eval_scores, final_latents):
        whatsoever_key = list(final_latents.keys())[0]
        cur_batch_size = final_latents[whatsoever_key].shape[0]

        subdir = os.path.join(self.save_root, "stats")
        if (not os.path.exists(subdir)): os.makedirs(subdir)

        for i in range(cur_batch_size):
            sample_id = self.cur_global_id + i
            pkl_path = os.path.join(subdir, str(sample_id).zfill(6)+".pkl")

            stats = {}
            if eval_scores is not None:
                stats["eval"] = {}
                for k,v in eval_scores.items():
                    if v is not None:
                        if v.ndim == 0:
                            stats["eval"][k] = v.item()
                        else:
                            stats["eval"][k] = v[i].item()
            
            stats["latents"] = {}
            for k,v in final_latents.items():
                if isinstance(v, list):
                    stats["latents"][k] = [vv[i].detach().cpu() for vv in v]
                elif isinstance(v, torch.Tensor):
                    stats["latents"][k] = v[i].detach().cpu()
                else:
                    assert v is None, \
                        "Got unsaved latents {}".format(k)

            pkl.dump(stats, open(pkl_path, "wb"))

    def _pt_to_img(self, tensor):
        tensor = tensor.clamp(-1, 1).permute(0, 2, 3, 1)
        tensor = (tensor + 1) / 2
        tensor = tensor.detach().cpu().numpy()
        return tensor

    def _loss_agg(self, losses, progress_ratio):
        total_loss = 0
        for name,loss in losses.items():
            try:
                loss_weight = getattr(self.config.task.losses_weights, name)
            except AttributeError as e:
                print("Cannot find loss weighting definition of {} within the config!".format(name))
                raise e
            if hasattr(self.config.task, "losses_weight_progress"):
                if hasattr(self.config.task.losses_weight_progress, name):
                    progress_factor = self._loss_weight_progress(
                        getattr(self.config.task.losses_weight_progress, name), progress_ratio)
                    loss_weight *= progress_factor
            total_loss += (loss_weight * loss)
        return total_loss

    def _loss_weight_progress(self, manipulation_type, progress_ratio):
        if manipulation_type == "linear-decay":
            return (1 - progress_ratio)
        elif "disable-at-" in manipulation_type:
            point = float(manipulation_type.split("-")[-1])
            assert point < 1+1e-6, "Please use ratio."
            return 0 if progress_ratio > point else 1
        else:
            raise NotImplementedError("Unknown manipulation type {}".format(manipulation_type))
    
    @torch.no_grad()
    def _create_trait_variations(self, ss_global_latents, ss_local_latents, coords, ts_styles, ts_noises, 
                                 inv_raw, full_structure_latent_shape, target_imgs, center_drift):

        # Create images with random variables, see which var is responsible for what visual part
        NUM_VAR_PER_TRAIT = 2

        outputs = [inv_raw.detach().cpu()]
        cur_batch_size = ss_local_latents.shape[0]

        # Random SS global
        for v_id in range(NUM_VAR_PER_TRAIT):
            tmp_ss_global = self.latent_sampler.sample_global_latent(
                cur_batch_size, 
                mixing=False, 
                device=self.device,
                requires_grad=False)
            inter_output = self.g_ema(
                global_latent=tmp_ss_global,
                local_latent=ss_local_latents,
                override_coords=coords,
                early_return_structure_latent=True,
                disable_dual_latents=True)
            output = self.g_ema(
                styles=ts_styles, 
                structure_latent=inter_output["structure_latent"],
                override_coords=coords,
                noises=ts_noises,
                disable_dual_latents=True).detach().cpu()
            inv_raw = self.output_size_coorection(output["gen"])
            inv_results = inv_raw #self._crop_aligned_region(inv_raw, target_imgs, center_drift)
            outputs.append(inv_results)

        # Random SS local
        for v_id in range(NUM_VAR_PER_TRAIT):
            tmp_ss_local_latents = self.latent_sampler.sample_local_latent(
                cur_batch_size, 
                device=self.device,
                requires_grad=False,
                specific_shape=full_structure_latent_shape)
            inter_output = self.g_ema(
                global_latent=ss_global_latents,
                local_latent=tmp_ss_local_latents,
                override_coords=coords,
                early_return_structure_latent=True,
                disable_dual_latents=True)
            output = self.g_ema(
                styles=ts_styles, 
                structure_latent=inter_output["structure_latent"],
                override_coords=coords,
                noises=ts_noises,
                disable_dual_latents=True).detach().cpu()
            inv_raw = self.output_size_coorection(output["gen"])
            inv_results = inv_raw #self._crop_aligned_region(inv_raw, target_imgs, center_drift)
            outputs.append(inv_results)

        # Random TS styles
        for v_id in range(NUM_VAR_PER_TRAIT):
            tmp_global_latents = self.latent_sampler.sample_global_latent(
                cur_batch_size, 
                device=self.device) # (B, 2, C)
            # tmp_ts_styles = \
            #     self.g_ema.texture_synthesizer.get_style(tmp_global_latents[:, 0])
            inter_output = self.g_ema(
                global_latent=ss_global_latents,
                local_latent=ss_local_latents,
                override_coords=coords,
                early_return_structure_latent=True,
                disable_dual_latents=True)
            output = self.g_ema(
                global_latent=tmp_global_latents, 
                structure_latent=inter_output["structure_latent"],
                override_coords=coords,
                noises=ts_noises,
                disable_dual_latents=True).detach().cpu()
            inv_raw = self.output_size_coorection(output["gen"])
            inv_results = inv_raw #self._crop_aligned_region(inv_raw, target_imgs, center_drift)
            outputs.append(inv_results)

        # Random TS noises
        for v_id in range(NUM_VAR_PER_TRAIT):
            tmp_ts_noises = [
                torch.randn_like(n, device=self.device) for n in ts_noises]
            inter_output = self.g_ema(
                global_latent=ss_global_latents,
                local_latent=ss_local_latents,
                override_coords=coords,
                early_return_structure_latent=True,
                disable_dual_latents=True)
            output = self.g_ema(
                styles=ts_styles, 
                structure_latent=inter_output["structure_latent"],
                override_coords=coords,
                noises=tmp_ts_noises,
                disable_dual_latents=True).detach().cpu()
            inv_raw = self.output_size_coorection(output["gen"])
            inv_results = inv_raw #self._crop_aligned_region(inv_raw, target_imgs, center_drift)
            outputs.append(inv_results)

        B, C, H, W = outputs[0].shape
        nrows = len(outputs)
        outputs = make_grid(
            torch.stack(outputs, 1).reshape(nrows*B, C, H, W), 
            nrow=nrows, normalize=True, range=(-1, 1))

        return outputs

    def get_future_coords(self, ss_local_latents):
        # Mostly copied from `test_managers.infinite_generation.py:_init_starting_points()` 
        #   and `test_managers.testing_vars_wrapper.py:replace_by_records()`, basically do the same things
        future_target_height, future_target_width = self.config.task.future_resolution
        future_meta_height, future_meta_width = self._calc_future_meta_shape(
            self.config.task.future_resolution[0], self.config.task.future_resolution[1])
        future_structure_latent_size = [
            self.g_ema_module.calc_in_spatial_size(future_meta_height, include_ss=False),
            self.g_ema_module.calc_in_spatial_size(future_meta_width, include_ss=False)]
        future_local_latent_size = [
            future_structure_latent_size[0] + self.ss_unfold_size * 2,
            future_structure_latent_size[1] + self.ss_unfold_size * 2,
        ]
        future_meta_coords = self.coord_handler.sample_coord_grid(
            spatial_latent=None,
            specific_shape=future_local_latent_size, 
            device=self.device,
            batch_size=ss_local_latents.shape[0],
            is_training=False)
        future_meta_pad_h = (future_meta_height - future_target_height) // 2
        future_meta_pad_w = (future_meta_width  - future_target_width) // 2
        future_img_c_loc_pix = [
            round(self.config.task.future_placement[0] * future_target_height + future_meta_pad_h),
            round(self.config.task.future_placement[1] * future_target_width + future_meta_pad_w)]
        mock_meta_img = torch.zeros(1, 1, future_meta_height, future_meta_width)
        _, _, pin_loc_list_ss, _ = \
            self.g_ema_module.calibrate_spatial_shape(
                mock_meta_img, direction="backward", padding_mode="replicate", 
                verbose=False, pin_loc=future_img_c_loc_pix)
        future_coord_c_loc_pix = pin_loc_list_ss[0]
        future_coord_st_loc_pix = [
            future_coord_c_loc_pix[0] - (ss_local_latents.shape[2] // 2),
            future_coord_c_loc_pix[1] - (ss_local_latents.shape[3] // 2),
        ]
        coords = future_meta_coords[
            :, 
            :, 
            future_coord_st_loc_pix[0]:future_coord_st_loc_pix[0]+ss_local_latents.shape[2],
            future_coord_st_loc_pix[1]:future_coord_st_loc_pix[1]+ss_local_latents.shape[3],
        ]
        assert future_coord_st_loc_pix[0] >=0 , \
            "Coordinate selection out of X bound 0 (got {})".format(
                future_coord_st_loc_pix[0])
        assert future_coord_st_loc_pix[0]+ss_local_latents.shape[2] <= future_meta_coords.shape[2], \
            "Coordinate selection out of X bound {} (got {})".format(
                future_meta_coords.shape[2], 
                future_coord_st_loc_pix[0]+ss_local_latents.shape[2])
        assert future_coord_st_loc_pix[1] >=0 , \
            "Coordinate selection out of Y bound 0 (got {})".format(
                future_coord_st_loc_pix[1])
        assert future_coord_st_loc_pix[1]+ss_local_latents.shape[3] <= future_meta_coords.shape[3], \
            "Coordinate selection out of Y bound {} (got {})".format(
                future_meta_coords.shape[3], 
                future_coord_st_loc_pix[1]+ss_local_latents.shape[3])
        # print(" [*] To the future, the coords are will be placed at [NE] ({}, {}) with shape {}.".format(
        #     future_coord_st_loc_pix[0], future_coord_st_loc_pix[1], coords.shape))
        return coords

    def _calc_future_meta_shape(self, future_height, future_width):
        # Get all the starting points of the patches to generate
        num_steps_h = \
            math.ceil((future_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        num_steps_w = \
            math.ceil((future_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        future_meta_height = self.pixelspace_step_size * (num_steps_h-1) + self.outfeat_sizes_list[-1]
        future_meta_width  = self.pixelspace_step_size * (num_steps_w-1) + self.outfeat_sizes_list[-1]
        return future_meta_height, future_meta_width

    
    def output_size_coorection(self, img):
        pad_h = (self.meta_height - self.target_height) // 2
        pad_w = (self.meta_width - self.target_width) // 2
        return img[:, :, pad_h:pad_h+self.target_height, pad_w:pad_w+self.target_width]

            



