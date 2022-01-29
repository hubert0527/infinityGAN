import os
import yaml
import random
import shutil
import argparse
import traceback
import numpy as np
from PIL import Image, ImageFont, ImageDraw 
from tqdm import tqdm
from glob import glob
from easydict import EasyDict
from tensorboardX import SummaryWriter

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import utils

from dataset import MultiResolutionDataset, DictTensor
from models.losses import (
    d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize, coord_ac_loss)
from libs.fid import eval_fid
from libs.calc_inception import load_patched_inception_v3
from libs.backup import backup_files

from latent_sampler import LatentSampler
from utils import *


SET_TEST_ID = True # Making testing-time randomized noise inputs from StyleGAN2 fixed
TB_PARTITION_STEPS = 100000 # Partition event file into multiple chunks for efficient rsync


def train(args, loaders, latent_sampler, generator, discriminator, 
          g_optim, d_optim, g_ema, inception, device):
    global backup_files
        
    """
    Setup env
    """
    exp_root = os.path.join(config.var.log_dir, config.var.exp_name)
    if args.debug:
        # Do not write anything to disk
        torch.save = dummy_func
        backup_files = dummy_func
        ckpt_root = backup_root = ""
        writer = IdleWriter()
    else:
        ckpt_root = os.path.join(exp_root, "ckpt")
        if not os.path.exists(ckpt_root): os.makedirs(ckpt_root)
        backup_root = os.path.join(exp_root, "codes")
        if (not args.archive_mode):
            backup_files(cur_dir=os.getcwd(), backup_dir=backup_root)
            try:
                shutil.copy2(args.config, os.path.join(exp_root, os.path.basename(args.config)))
            except shutil.SameFileError:
                pass
        writer = SummaryWriter(os.path.join("logs", config.var.exp_name))
        

    if config.var.dataparallel:
        g_module = generator.module
        d_module = discriminator.module
        g_ema_module = g_ema.module
        ext_inf_device = "cuda"
    else:
        g_module = generator
        d_module = discriminator
        g_ema_module = g_ema
        ext_inf_device = "cuda"
    ext_inf_module = g_ema_module #.to(ext_inf_device)

    
    font_file_path = find_font_source()
    if font_file_path is None: 
        print("Cannot find font file, some logging items is omitted!")


    """
    Default values
    """
    iter_ = 0 # Placeholder for contextual functions
    if config.var.mean_path_lengths is None:
        mean_path_lengths = [torch.tensor(0.0, device=device)]
    else:
        mean_path_lengths = [v.to(device) for v in config.var.mean_path_lengths]
    losses = {}
    distrs = {}
    grad_norms = {}


    """
    Additional configurations
    """
    accum = 0.5 ** (32 / (10 * 1000))
    grad_logging_excludes = ["modulation", "noise", "bias", "const"]

    # Adjust testing batching
    n_test_batches = config.log_params.n_save_sample // config.train_params.batch_size
    n_test_samples = max(1, n_test_batches * config.train_params.batch_size)
    if n_test_samples != config.log_params.n_save_sample:
        print(" [!] Previous N-testing samples {} is not good for visualization, adjust to {}!".format(
            config.log_params.n_save_sample, n_test_samples))
        config.log_params.n_save_sample = n_test_samples
        n_test_batches = n_test_samples // config.train_params.batch_size # Still may different due to the `max()`

    """
    Create fixed latents for visualization
    """
    vis_local_latent = latent_sampler.sample_local_latent(config.log_params.n_save_sample, device)
    vis_global_latent = latent_sampler.sample_global_latent(config.log_params.n_save_sample, device, mixing=False)
    vis_test_ids = torch.arange(config.log_params.n_save_sample) if SET_TEST_ID else None
    vis_inject_index = random.randint(1, g_ema_module.texture_synthesizer.n_latent - 1)

    """
    Create extended latents for visualization
    """
    if config.train_params.patch_size > 512:
        ext_mult_list = [] # Consumes too much mem, do not log ext images during training
    elif config.train_params.patch_size > 256:
        ext_mult_list = [2,]
    else:
        ext_mult_list = [2, 4]
    vis_local_latent_ext_list = [
        latent_sampler.sample_local_latent(
            config.log_params.n_save_sample, device, spatial_size_enlarge=mult) 
        for mult in ext_mult_list]
    # Make larger local latent contain the smaller local latent in the center
    if len(vis_local_latent_ext_list) > 0:
        vis_local_latent, vis_local_latent_ext_list = unify_local_latent_ext(vis_local_latent, vis_local_latent_ext_list)


    """
    Create sample generation function for FID calculation
    """
    def generation_fn(n_batches):
        for i in range(n_batches):
            global_latent = latent_sampler.sample_global_latent(config.train_params.batch_size, device)
            local_latent = latent_sampler.sample_local_latent(config.train_params.batch_size, device)
            gen_data = g_ema(global_latent=global_latent, 
                             local_latent=local_latent, 
                             is_fid_eval=True,
                             disable_dual_latents=True)
            yield gen_data["gen"]

    def generation_fn_ext2(n_batches):
        for i in range(n_batches):
            gen_data_list = []
            for j in range(config.train_params.batch_size):
                global_latent = latent_sampler.sample_global_latent(1, ext_inf_device)
                local_latent = latent_sampler.sample_local_latent(1, ext_inf_device, spatial_size_enlarge=2)
                gen_data = ext_inf_module(global_latent=global_latent, 
                                          local_latent=local_latent,
                                          is_fid_eval=True,
                                          disable_dual_latents=True).detach()
                patch = gen_data["gen"]

                full_size = config.train_params.full_size
                gen_size = patch.shape[-1]
                # EXT2 generation resolution should be equal or larger than the full image size
                if gen_size > full_size:
                    pad = (gen_size-full_size) // 2
                    patch = patch[:, :, pad:pad+full_size, pad:pad+full_size]
                gen_data_list.append(patch)
            yield torch.cat(gen_data_list, 0)


    """
    Check if the dimension is correct
    """
    if config.train_params.ts_no_zero_pad:
        expected_structure_latent_spatial_size = g_module.calc_in_spatial_size(
            config.train_params.patch_size, include_ss=False)
        assert expected_structure_latent_spatial_size == config.train_params.ts_input_size, \
            "Expects structure_latent shape is {}, but got {} in the config!".format(
                expected_structure_latent_spatial_size, config.train_params.ts_input_size)


    """
    Start training
    """
    pbar = range(config.var.start_iter, config.train_params.iter)
    pbar = tqdm(pbar, initial=config.var.start_iter, total=config.train_params.iter, 
                dynamic_ncols=True, smoothing=0.01)

    for iter_ in pbar:

        discriminator.train()

        real_data = next(loaders["train"])
        real_data = DictTensor(real_data)
        if config.var.dataparallel: 
            pass # Dataparallel splits data partition in the backend
        else:
            real_data = real_data.to(device)

        
        """
        Train D
        """
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        global_latent = latent_sampler.sample_global_latent(config.train_params.batch_size, device)
        local_latent = latent_sampler.sample_local_latent(config.train_params.batch_size, device)
        fake_data = generator(
            global_latent=global_latent, 
            local_latent=local_latent)

        fake_pred = discriminator(fake_data)
        real_pred = discriminator(real_data)
        if config.train_params.training_modality == "patch":
            d_loss = d_logistic_loss(real_pred["d_patch"], fake_pred["d_patch"])
            losses["d_adv_loss"] = d_loss.detach().cpu()
        else:
            raise NotImplementedError()

        if hasattr(config.train_params, "coord_use_ac") and config.train_params.coord_use_ac:
            d_coord_ac_real = coord_ac_loss(
                real_pred["ac_coords_pred"], real_data["ac_coords"], "real", config)
            d_coord_ac_fake = coord_ac_loss(
                fake_pred["ac_coords_pred"], fake_data["ac_coords"], "fake", config)
            losses["d_ac_coords_real"] = d_coord_ac_real.detach().cpu()
            losses["d_ac_coords_fake"] = d_coord_ac_fake.detach().cpu()
            d_loss = d_loss + (d_coord_ac_real + d_coord_ac_fake) * config.train_params.coord_ac_w
            d_coord_ac_real, d_coord_ac_fake = None, None

            distrs["real_ac_pred_x"] = real_pred["ac_coords_pred"][:, 0].detach().cpu()
            distrs["fake_ac_pred_x"] = fake_pred["ac_coords_pred"][:, 0].detach().cpu()

        losses["d_total_loss"] = d_loss.detach().cpu()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        # Logging
        losses["d"] = d_loss.detach().cpu()
        if "ac_coords" in real_data:
            distrs["real_ac_coords_x"] = real_data["ac_coords"][:, 0].detach().cpu()
            if real_data["ac_coords"].shape[1] > 1:
                distrs["real_ac_coords_y"] = real_data["ac_coords"][:, 1].detach().cpu()
        if "ac_coords" in fake_data:
            distrs["fake_ac_coords_x"] = fake_data["ac_coords"][:, 0].detach().cpu()
            if fake_data["ac_coords"].shape[1] > 1:
                distrs["fake_ac_coords_y"] = fake_data["ac_coords"][:, 1].detach().cpu()


        """
        D R1 regularization
        """
        if iter_ % config.train_params.d_reg_every == 0:
            real_data.requires_grad = True
            purge_cache() # Try to solve memory-leak problem

            real_pred = discriminator(real_data)
            if config.train_params.training_modality == "patch":
                r1_loss = d_r1_loss(real_pred["d_patch"], real_data["patch"])
                compute_node = real_pred["d_patch"][0]
            else:
                raise NotImplementedError()

            if config.var.dataparallel:
                compute_node = compute_node.cpu()
            discriminator.zero_grad()
            (config.train_params.r1 / 2 * r1_loss * config.train_params.d_reg_every + 0 * compute_node).backward()
            d_optim.step()
            losses["r1"] = r1_loss.detach().cpu()
            del r1_loss, compute_node
        elif "r1" not in losses:
            losses["r1"] = torch.tensor(0.0, device=device)
            

        """
        Train Generator
        """
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        global_latent = latent_sampler.sample_global_latent(config.train_params.batch_size, device)
        local_latent = latent_sampler.sample_local_latent(config.train_params.batch_size, device)
        fake_data = generator(
            global_latent=global_latent, 
            local_latent=local_latent)

        fake_pred = discriminator(fake_data)
        if config.train_params.training_modality == "patch":
            g_loss = g_nonsaturating_loss(fake_pred["d_patch"])
            losses["g_adv_loss"] = g_loss.detach().cpu()
        else:
            raise NotImplementedError()
            
        if hasattr(config.train_params, "coord_use_ac") and config.train_params.coord_use_ac:
            g_coord_ac_fake = coord_ac_loss(
                fake_pred["ac_coords_pred"], fake_data["ac_coords"], "fake", config)
            losses["g_ac_coords_fake"] = g_coord_ac_fake.detach().cpu()
            g_loss = g_loss + g_coord_ac_fake * config.train_params.coord_ac_w
            g_coord_ac_fake = None

        if hasattr(config.train_params, "diversity_z_w") and config.train_params.diversity_z_w!=0:
            fake_data["diversity_z_loss"] = fake_data["diversity_z_loss"].mean()
            losses["diversity_z_loss"] = fake_data["diversity_z_loss"].detach().cpu()
            g_loss += fake_data["diversity_z_loss"] * config.train_params.diversity_z_w

        losses["g_total_loss"] = g_loss.detach().cpu()

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # Track grads, log less frequently
        if (iter_ % config.log_params.img_tick == 0) or args.debug:
            if hasattr(g_module, "structure_synthesizer") and (g_module.structure_synthesizer is not None):
                for name,params in g_module.structure_synthesizer.named_parameters():
                    if any([k in name for k in grad_logging_excludes]): continue
                    distrs["SS-grad/" + name + "-grad"] = params.grad.detach().cpu()
                    grad_norms["SS-grad-norm/" + name + "-grad"] = params.grad.detach().norm().cpu().item()
            for name,params in g_module.texture_synthesizer.named_parameters():
                if any([k in name for k in grad_logging_excludes]): continue
                distrs["TS-grad/" + name + "-grad"] = params.grad.detach().cpu()
                grad_norms["TS-grad-norm/" + name + "-grad"] = params.grad.detach().norm().cpu().item()
                
        del g_loss, fake_pred, fake_data


        """
        G path regularization
        """
        if iter_ % config.train_params.g_reg_every == 0:
            purge_cache() # Try to solve memory-leak problem

            path_batch_size = max(1, config.train_params.batch_size // config.train_params.path_batch_shrink)
            global_latent = latent_sampler.sample_global_latent(path_batch_size, device)
            local_latent = latent_sampler.sample_local_latent(path_batch_size, device)
            fake_data = generator(
                global_latent=global_latent, 
                local_latent=local_latent,
                return_path_length=True)

            path_loss, mean_path_lengths = \
                g_path_regularize(fake_data["path_lengths"], mean_path_lengths)

            generator.zero_grad()
            weighted_path_loss = config.train_params.path_regularize * config.train_params.g_reg_every * path_loss
            weighted_path_loss.backward()
            g_optim.step()

            losses["path"] = path_loss.detach().cpu()
            losses["path_lengths"] = \
                torch.stack([v.mean().detach().cpu() for v in fake_data["path_lengths"]]).mean()
            del path_loss, fake_data, weighted_path_loss
        else:
            losses["path"] = torch.tensor(0.0, device=device)
            losses["path_lengths"] = torch.tensor(0.0, device=device)

        """
        Accumlate G
        """
        accumulate(g_ema_module, g_module, accum)

        """
        Logging
        """
        with torch.no_grad():
            d_total_loss_val = losses["g_total_loss"].mean().item()
            g_total_loss_val = losses["d_total_loss"].mean().item()
            r1_val           = losses["r1"].mean().item()
            path_loss_val    = losses["path"].mean().item()
            path_length_val  = losses["path_lengths"].mean().item()
            mean_path_lengths_val = torch.stack(mean_path_lengths).mean().item()

            pbar.set_description("; ".join([
                f" [*] EXP: {config.var.exp_name}", 
                f"d: {d_total_loss_val:.2f}",
                f"g: {g_total_loss_val:.2f}",
                f"r1: {r1_val:.2f}; ",
                f"mean path: {mean_path_lengths_val:.2f}",
            ]))
                

            if iter_ % config.log_params.log_tick == 0:
                writer.add_scalar("losses/d_total_loss", d_total_loss_val, iter_)
                writer.add_scalar("losses/g_total_loss", g_total_loss_val, iter_)
                writer.add_scalar("losses/r1", r1_val, iter_)
                writer.add_scalar("losses/path_loss", path_loss_val, iter_)

                writer.add_scalar("utils/path_lengths", path_length_val, iter_)
                writer.add_scalar("utils/mean_path_lengths", mean_path_lengths_val, iter_)

                breakdown_tracking_list = [
                    "g_adv_loss", "d_adv_loss", "diversity_z_loss", 
                    "d_ac_coords_real", "d_ac_coords_fake", "g_ac_coords_fake"]
                for k in breakdown_tracking_list:
                    if k in losses:
                        writer.add_scalar("losses_breakdown/{}".format(k), losses[k].mean().item(), iter_)

                for k in distrs:
                    try:
                        writer.add_histogram(k, distrs[k].numpy(), iter_, bins=1000)
                    except Exception as e:
                        print(" [!] Error happens on distr {}".format(k))

                log_memory(config, writer, iter_)


            if (iter_ % config.log_params.img_tick == 0) or args.debug:
                for k in grad_norms:
                    writer.add_scalar(k, grad_norms[k], iter_)

            
            # [Visualize] Real samples, only write at the beginning 
            if iter_ == 0:
                nrow_normal = int(config.log_params.n_save_sample ** 0.5)
                meta_img = utils.make_grid(
                    real_data["patch"],
                    nrow=nrow_normal,
                    normalize=True,
                    range=(-1, 1),
                )
                writer.add_image("real", meta_img, iter_)


            is_img_tick = (iter_ % config.log_params.img_tick == 0) and (iter_ != config.var.start_iter)
            if is_img_tick or args.debug:
                g_ema.eval()
                discriminator.eval()

                # [Visualize] Random samples
                sample = auto_batched_inference(
                    g_ema, config,
                    partition_keys=["global_latent", "local_latent", "test_ids"],
                    global_latent=vis_global_latent, 
                    local_latent=vis_local_latent, 
                    test_ids=vis_test_ids,
                    disable_dual_latents=True,
                    inject_index=vis_inject_index)
                meta_img = utils.make_grid(
                    sample["gen"],
                    nrow=int(config.log_params.n_save_sample ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )
                writer.add_image("rnd_gen", meta_img, iter_)
                del sample

                # [Visualize] Random samples with structural diversity, same global latent with different local latents
                group_size = config.train_params.batch_size // 2
                ori_shape = vis_global_latent.shape
                sample = auto_batched_inference(
                    g_ema,
                    config,
                    partition_keys=["global_latent", "local_latent", "test_ids"],
                    global_latent=vis_global_latent[::group_size].unsqueeze(1).repeat(1, group_size, 1, 1).reshape(*ori_shape),
                    local_latent=vis_local_latent, 
                    test_ids=vis_test_ids,
                    disable_dual_latents=True,
                    inject_index=vis_inject_index)
                meta_img = utils.make_grid(
                    sample["gen"],
                    nrow=group_size,
                    normalize=True,
                    range=(-1, 1),
                )
                writer.add_image("structure_diversity", meta_img, iter_)


                # [Visualize] The D prediction of the the samples
                if font_file_path is not None:
                    pred = discriminator(sample.to(device))["d_patch"]
                    min_, max_ = pred.min(), pred.max()
                    pred = (pred - min_) / (max_ - min_) # normalize to (0, 1)
                    pred = pred * 0.8 + 0.1
                    imgs = []
                    for i in range(pred.shape[0]):
                        score = round(pred[i].item(), 2)
                        text_c = 255 if score<0.5 else 0

                        img = Image.fromarray((np.ones((64, 64, 3))*255*pred[i].item()).astype(np.uint8))
                        draw = ImageDraw.Draw(img)
                        font = ImageFont.truetype("./assets/OpenSans-Bold.ttf", 22)
                        draw.text((7, 16), "{: .2f}".format(score), (text_c, text_c, text_c), font=font)
                        imgs.append(np.array(img))
                    imgs = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)).float()
                    meta_img = utils.make_grid(
                        imgs,
                        nrow=int(config.log_params.n_save_sample ** 0.5),
                        normalize=True,
                        range=(0, 255),
                    )
                    writer.add_image("structure_diversity_D_score", meta_img, iter_)

                    del pred, imgs
                del sample


                # [Visualize] Random samples with style diversity, same local latent with different global latents
                group_size = config.train_params.batch_size // 2
                n_groups = vis_global_latent.shape[0] // group_size
                ori_shape = vis_global_latent.shape
                vis_test_ids_grouped = vis_test_ids[:n_groups].view(n_groups,1).repeat(1,group_size).view(-1) if vis_test_ids is not None else vis_test_ids
                outputs = auto_batched_inference(
                    g_ema,
                    config,
                    partition_keys=["global_latent", "local_latent"],
                    global_latent=vis_global_latent,
                    local_latent=vis_local_latent, 
                    early_return_structure_latent=True,
                    disable_dual_latents=True,
                    inject_index=vis_inject_index)
                if isinstance(outputs, DictTensor):
                    structure_latent = outputs["structure_latent"] # s or not s, such a good question...
                    coords = outputs["coords"]
                else:
                    structure_latent = outputs
                    coords = None
                spatial_shape = structure_latent.shape # (BxCxHxW)
                sample = auto_batched_inference(
                    g_ema,
                    config,
                    partition_keys=["global_latent", "structure_latent", "override_coords", "test_ids"],
                    global_latent=vis_global_latent,
                    structure_latent=structure_latent[::group_size].unsqueeze(1).repeat(1, group_size, 1, 1, 1).reshape(*spatial_shape).to(device),
                    override_coords=coords.to(device) if coords is not None else None, # was cast to cpu in auto_batched_inference
                    disable_dual_latents=True,
                    test_ids=vis_test_ids_grouped,
                    inject_index=vis_inject_index)
                meta_img = utils.make_grid(
                    sample["gen"],
                    nrow=group_size,
                    normalize=True,
                    range=(-1, 1),
                )
                writer.add_image("style_diversity", meta_img, iter_)
                del sample


                # [Visualize] Random samples with larger spatial size
                for ext_mult,vis_local_latent_ext in zip(ext_mult_list,vis_local_latent_ext_list):
                    nrow_normal = int(config.log_params.n_save_sample ** 0.5)
                    nrow_ext = max(3, nrow_normal // ext_mult) if nrow_normal > 3**2 else nrow_normal
                    samples = []
                    for i in range(nrow_ext**2):
                        sample = ext_inf_module(
                            global_latent=vis_global_latent[i:i+1].to(ext_inf_device), 
                            local_latent=vis_local_latent_ext[i:i+1].to(ext_inf_device), 
                            test_ids=[i] if SET_TEST_ID else None,
                            disable_dual_latents=True,
                            inject_index=vis_inject_index).detach().cpu()
                        samples.append(sample["gen"])

                        """ Really in some cases.
                        if i==0 and ext_mult==ext_mult_list[-1] and (font_file_path is not None):
                            cidx = np.random.randint(0, sample["structure_latent"].shape[1]-1)
                            meta_img = vis_structure_latent_slice(sample["structure_latent"][0,cidx])
                            writer.add_image("structure_latent_slice_ext", meta_img, iter_)
                        """
                            
                    samples = torch.cat(samples, 0)
                    meta_img = utils.make_grid(
                        samples,
                        nrow=nrow_ext,
                        normalize=True,
                        range=(-1, 1),
                    )
                    writer.add_image("rnd_gen_ext{}".format(ext_mult), meta_img, iter_)

                purge_cache()
                                

            if (iter_ % config.log_params.save_tick == 0) and (iter_ > config.var.start_iter):
                torch.save({
                    "iter": iter_,
                    "best_fid": config.var.best_fid,
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_ema": g_ema_module.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "mean_path_lengths": [v.detach().cpu() for v in mean_path_lengths],
                }, os.path.join(ckpt_root, "inter_{}.pth.tar".format(str(iter_).zfill(8))))
                rm_outdated_ckpt(pattern=os.path.join(ckpt_root, "inter_*.pth.tar"), max_to_keep=2)


            """
            Vanilla FID (patch)
            """
            if config.test_params.calc_fid:
                if ((iter_ % config.log_params.eval_tick == 0) and (iter_ > config.var.start_iter)) or args.debug: 
                    stats_key = "{}-{}".format(config.data_params.dataset, config.train_params.patch_size)
                    # FID statistics can be different for different PyTorch version, not sure about cuda
                    stats_key += f"_PT{torch.__version__}_cu{torch.version.cuda}"
                    g_ema.eval()
                    if (iter_ == 0) and (not args.debug): 
                        # Create graph only, not really calculating
                        fid = eval_fid(loaders["fid-train"], generation_fn, inception, stats_key, "patch", device, config, no_write_cache=args.debug, create_graph_only=True)
                    else:
                        fid = eval_fid(loaders["fid-train"], generation_fn, inception, stats_key, "patch", device, config, no_write_cache=args.debug)
                    print(" [*] FID = {} (best FID = {})".format(fid, config.var.best_fid))

                    writer.add_scalar("metrics/fid_train", fid, iter_)
                    if fid < config.var.best_fid:
                        config.var.best_fid = fid
                        torch.save({
                            "iter": iter_,
                            "best_fid": config.var.best_fid,
                            "g": g_module.state_dict(),
                            "d": d_module.state_dict(),
                            "g_ema": g_ema_module.state_dict(),
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                            "mean_path_lengths": [v.detach().cpu() for v in mean_path_lengths],
                        }, os.path.join(ckpt_root, "best_fid.pth.tar".format(str(iter_).zfill(8))))
                    purge_cache()


            """
            Ext2 FID

            This one is much~ slower, run less frequently.
            """
            if config.test_params.calc_fid_ext2:
                is_tick = (iter_ % config.log_params.fid_ext2_tick == 0) and (iter_ > config.var.start_iter)
                is_computable = config.train_params.full_size > config.train_params.patch_size
                if ((is_tick and is_computable) or args.debug):
                    stats_key = "{}-{}-full".format(config.data_params.dataset, config.train_params.full_size)
                    # FID statistics can be different for different PyTorch version, not sure about cuda
                    stats_key += f"_PT{torch.__version__}_cu{torch.version.cuda}"
                    if (iter_ == 0) and (not args.debug): 
                        ext2_fid = 500 # Do not waste time...
                    else:
                        ext2_fid = eval_fid(loaders["fid-train"], generation_fn_ext2, inception, stats_key, "full", device, config, no_write_cache=args.debug)
                    print(" [*] EXT2 FID = {}".format(ext2_fid))
                    writer.add_scalar("metrics/ext2_fid_train", ext2_fid, iter_)
                    purge_cache()

            if args.debug:
                exit()
        
        if iter_ > config.var.start_iter and iter_ % TB_PARTITION_STEPS == 0:
            writer.close()
            writer = SummaryWriter(os.path.join("logs", config.var.exp_name))

    print("Done!")
    exit()


if __name__ == "__main__":
    
    try:

        parser = argparse.ArgumentParser()
        parser.add_argument("config", type=str)
        parser.add_argument("--debug", action="store_true", default=False)
        parser.add_argument("--archive-mode", action="store_true", default=False)
        parser.add_argument("--clear-fid-cache", action="store_true", default=False)
        parser.add_argument("--num-workers", type=int, default=16)
        args = parser.parse_args()

        fid_cache_dir = ".fid-cache/"
        if args.clear_fid_cache and os.path.exists(fid_cache_dir):
            os.rmtree(fid_cache_dir)
        
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config = EasyDict(config)
            config.var = EasyDict()
        config.var.exp_name = os.path.basename(args.config).split(".yaml")[0]
        print(" [*] Config {} loaded!".format(args.config))

        if args.archive_mode:
            config.var.log_dir = "../../" # Running in ./logs/<exp_name>/codes/
        else:
            config.var.log_dir = "./logs/"

        # Remove previous error file (will make confusion on log synchronizing)
        error_f = os.path.join(config.var.log_dir, config.var.exp_name, "error-log.txt")
        if os.path.exists(error_f):
            os.remove(error_f)

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            n_gpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            config.var.dataparallel = n_gpu > 1
            config.var.n_gpu = n_gpu
            if n_gpu > 1:
                torch.backends.cudnn.benchmark = True
            else:
                torch.backends.cudnn.benchmark = True
        else:
            raise ValueError(" [!] Please specify CUDA_VISIBLE_DEVICES!")

        # [NOTE] In debug mode:
        # 1. Will not write any logs
        # 2. Exit after first full iteration
        # 3. Force eval FID with one batch of fake samples; will not write FID cache if real stats are not exist
        if args.debug:
            print(" [Warning] Debug mode; Do not use this unless you know what you are doing!")
            bs = 4
            config.train_params.batch_size = bs * n_gpu
            config.log_params.n_save_sample = bs * n_gpu
            config.test_params.n_fid_sample = config.train_params.batch_size - 1
            print(" [Override] Setting training batch size to {} for faster debugging.".format(config.train_params.batch_size))
    

        """
        Build G & D
        """
        generator     = import_func(config.train_params.g_arch)(config=config)
        g_ema         = import_func(config.train_params.g_arch)(config=config)
        discriminator = import_func(config.train_params.d_arch)(config=config)

        """
        FID setup
        """
        inception = load_patched_inception_v3()
        inception.eval()

        """
        Multi-GPU
        """
        if config.var.dataparallel:
            device = "cpu" # torch will auto do the GPU partitioning in backend
            generator = nn.DataParallel(generator).cuda()
            discriminator = nn.DataParallel(discriminator).cuda()
            g_ema = nn.DataParallel(g_ema).cuda()
            inception = nn.DataParallel(inception).cuda()
            structure_synthesizer = generator.module.structure_synthesizer
            texture_synthesizer = generator.module.texture_synthesizer
        else:
            device = "cuda"
            generator = generator.to(device)
            discriminator = discriminator.to(device)
            g_ema = g_ema.to(device)
            inception = inception.to(device)
            structure_synthesizer = generator.structure_synthesizer
            texture_synthesizer = generator.texture_synthesizer


        """
        Create Optim
        """
        g_ema.eval()
        accumulate(g_ema, generator, 0)

        latent_sampler = LatentSampler(generator, config)

        g_reg_ratio = config.train_params.g_reg_every / (config.train_params.g_reg_every + 1)
        d_reg_ratio = config.train_params.d_reg_every / (config.train_params.d_reg_every + 1)

        g_optim = optim.Adam(
            list(generator.parameters()),
            lr=config.train_params.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        d_optim = optim.Adam(
            discriminator.parameters(),
            lr=config.train_params.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        """
        Load checkpoint
        """
        ckpt_dir = os.path.join(config.var.log_dir, config.var.exp_name, "ckpt")
        ckpts = sorted(glob(os.path.join(ckpt_dir, "inter_*.pth.tar")))
        if os.path.exists(ckpt_dir) and len(ckpts) > 0:
            latest_ckpt = ckpts[-1]
            print(" [*] Found ckpt, load model from:", latest_ckpt)

            ckpt = torch.load(latest_ckpt, map_location=lambda storage, loc: storage)

            config.var.start_iter = ckpt["iter"]
            config.var.best_fid = ckpt["best_fid"]
            if "mean_path_lengths" in ckpt:
                config.var.mean_path_lengths = ckpt["mean_path_lengths"]
            elif "mean_path_length" in ckpt: # backward compatibility
                config.var.mean_path_lengths = [
                    torch.tensor(ckpt["mean_path_length"])]
            else:
                config.var.mean_path_lengths = None # backward compatibilit
                print(" [!] Warning: Unless loading from prev ckpt, `mean_path_lengths` should be found in ckpt!")

            safe_load_state_dict(generator, ckpt["g"])
            safe_load_state_dict(discriminator, ckpt["d"])
            safe_load_state_dict(g_ema, ckpt["g_ema"])

            safe_load_state_dict(g_optim, ckpt["g_optim"])
            safe_load_state_dict(d_optim, ckpt["d_optim"])
        else:
            print(" [*] Did not find ckpt, fresh start!")
            config.var.start_iter = 0
            config.var.best_fid = 500
            config.var.mean_path_lengths = None 


        """
        Dataset
        """
        train_set = MultiResolutionDataset(
            split="train",
            config=config,
            is_training=True)
        valid_set = None
        #MultiResolutionDataset(
        #    os.path.join(dataset_root, "valid"), 
        #    is_training=False,
        #    config.train_params.full_size)
        train_set_fid = MultiResolutionDataset(
            split="train",
            config=config,
            is_training=False)

        loaders = {
            "train": make_nonstopping(data.DataLoader(
                train_set,
                batch_size=config.train_params.batch_size,
                sampler=data_sampler(train_set, shuffle=True),
                drop_last=True,
                num_workers=args.num_workers,
            )),
            #"valid": make_nonstopping(data.DataLoader(
            #    valid_set,
            #    batch_size=config.train_params.batch_size,
            #    sampler=data_sampler(valid_set, shuffle=False),
            #    drop_last=False,
            #)),
            "fid-train": data.DataLoader(
                train_set_fid,
                batch_size=config.train_params.batch_size,
                sampler=data_sampler(train_set_fid, shuffle=False),
                drop_last=False,
                num_workers=args.num_workers,
            ),
            #"fid-valid": data.DataLoader(
            #    valid_set,
            #    batch_size=config.train_params.batch_size,
            #    sampler=data_sampler(valid_set, shuffle=False),
            #    drop_last=False,
            #),
        }

        train(args, loaders, latent_sampler, generator, discriminator, g_optim, d_optim, g_ema, inception, device)
    except Exception as e:
        if e is not KeyboardInterrupt:
            error_dirs = sorted(glob("./burst-errors-*"))
            if args.debug:
                if len(error_dirs) > 0: # User run --debug via test_errors.py script
                    error_f = os.path.join(error_dirs[-1], "error-log-{}.txt".format(config.var.exp_name))
                else: # User run --debug by hand
                    error_f = os.path.join("error-log-{}.txt".format(config.var.exp_name))
            else: # Unexpected runtime errors
                error_f = os.path.join(config.var.log_dir, config.var.exp_name, "error-log.txt")
            with open(error_f, "w") as f:
                f.write(str(e) + "\n")
                f.write(" *** stack trace *** \n")
                f.write(traceback.format_exc())
        raise e
