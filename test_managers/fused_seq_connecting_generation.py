import os
import math
import datetime
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from torch._C import device
from tqdm import tqdm
from glob import glob
from itertools import product as iter_product

import torch

from models.ops import Blur
from test_managers.base_test_manager import BaseTestManager
from test_managers.testing_vars_wrapper import TestingVars
from test_managers.global_config import test_meta_extra_pad


class FusedSeqConnectingGenerationManager(BaseTestManager):

    def task_specific_init(self):

        self.local_latent_shape_single = (
            self.config.train_params.ts_input_size,
            self.config.train_params.ts_input_size,
        )
        self.local_latent_shape_connect = (
            self.config.train_params.ts_input_size,
            self.config.train_params.ts_input_size*2 + self.config.task.anchor_gap,
        )

        self.noise_heights_single = self.g_ema_module.calc_out_spatial_size(
            self.local_latent_shape_single[0], 
            include_ss=False, return_list=True)
        self.noise_widths_single = self.g_ema_module.calc_out_spatial_size(
            self.local_latent_shape_single[1], 
            include_ss=False, return_list=True)
        self.noise_widths_connect = self.g_ema_module.calc_out_spatial_size(
            self.local_latent_shape_connect[1], 
            include_ss=False, return_list=True)

        self.target_height = self.config.task.height
        self.target_width = self.noise_widths_connect[-1]
        self.anchor_patch_width = self.noise_widths_single[-1]
        self.fusion_patch_width = self.target_width - self.anchor_patch_width * 2
        self.ss_unfold_pad = self.config.train_params.ss_n_layers * self.config.train_params.ss_unfold_radius

        self.noise_gap_sizes = [b-a*2 for a,b in zip(self.noise_widths_single, self.noise_widths_connect)]

        self.meta_style_center_indices = [
            [
                center_f[0] * self.target_height,
                center_f[1] * self.target_width,
            ] for center_f in self.config.task.style_centers]

        self.blur_kernel = Blur(
            kernel=self.config.task.blur_kernel,
            pad=self.config.task.blur_kernel//2,
            padding_mode="replicate",
            prior=self.config.task.blur_type)

        # This is used to select which style to use at each spatial index
        print(" [*] Creating fusion map...")
        self.meta_fusion_map_ss, self.meta_fusion_map_ts = \
            self._create_fusion_map(self.meta_style_center_indices)


        # Create an initial stack of anchors
        assert len(self.config.task.style_centers) % 2 == 0, \
            "Expect an even number of anchors in the stack (so that we connect the middle two), but got {}".format(len(self.config.task.style_centers))

        self.output_stack = [] # The manager produces an anchor patch and a fusion patch at each step. Buffer the results to simulate an iterator.
        self.anchor_stack = []
        num_style_centers = len(self.config.task.style_centers)
        num_preceding_centers = num_style_centers // 2 - 1
        init_anchor_idx = num_style_centers - num_preceding_centers
        for i in range(-num_preceding_centers, init_anchor_idx):
            cur_anchor = self.create_vars(i)
            if len(self.anchor_stack) > 1: # Make the overlapped region (if happens) value consistent
                self.unify_overlapped_local_latent(self.anchor_stack[-1].local_latent, cur_anchor.local_latent)
            self.anchor_stack.append(cur_anchor)

        self.cur_anchor_idx = init_anchor_idx + 1

    def run_next(self, save=True, write_gpu_time=False, inv_records=None, inv_placements=None, calc_flops=False, disable_pbar=True, **kwargs):
        assert (inv_records is None) and (inv_placements is None), "Not implemented, we won't have that many inverted latents to inbetween."
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if v is not None:
                    print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))

        if (len(self.output_stack) > 0) and (self.config.task.save_type != "all-in-one"):
            cur_output = self.output_stack.pop(0)
            if save:
                self.save_meta_imgs(cur_output)
            return cur_output

        testing_vars = self.fuse_anchors(self.anchor_stack)
        self.generate(testing_vars, write_gpu_time=write_gpu_time, calc_flops=calc_flops)

        new_anchor = self.create_vars(self.cur_anchor_idx)
        self.unify_overlapped_local_latent(self.anchor_stack[-1].local_latent, new_anchor.local_latent)
        self.anchor_stack.append(new_anchor)
        self.anchor_stack = self.anchor_stack[1:]
        self.cur_anchor_idx += 1

        # We throw away the right anchor patch every step. 
        # It is possible to implement a more efficient function, but it will make the codes unreadable.
        anchor_patch = testing_vars.meta_img[:, :, :, :self.anchor_patch_width]
        fusion_patch = testing_vars.meta_img[:, :, :, self.anchor_patch_width:self.anchor_patch_width+self.fusion_patch_width]

        if self.config.task.save_type == "patches":
            cur_output = anchor_patch
            self.output_stack.append(fusion_patch)
        elif self.config.task.save_type == "patches-centercrop":
            cur_output = anchor_patch
            fusion_patch = self._center_crop(src=fusion_patch, tgt=anchor_patch)
            self.output_stack.append(fusion_patch)
        elif self.config.task.save_type == "anchors":
            cur_output = testing_vars.meta_img
        elif self.config.task.save_type == "all-in-one":
            save = False
            cur_output = None
            self.output_stack += [anchor_patch, fusion_patch]
        else:
            raise NotImplementedError("Unknown save type {}".format(self.config.task.save_type))
        
        if save and (cur_output is not None):
            self.save_meta_imgs(cur_output)
        return cur_output

    def create_vars(self, cur_anchor_idx):
        
        mixing = False
        assert mixing == False, "Otherwise, an injection index must be specified and fed into g_ema."

        # Allocate memory for the final output, starts iterating and filling in the generated results.
        # Can be reused
        meta_img = None # No need to create

        global_latent = self.latent_sampler.sample_global_latent(
            self.config.task.batch_size, mixing=mixing, device=self.device)
        ts_style = self.g_ema(
            call_internal_method="get_style", 
            internal_method_kwargs={"global_latent": global_latent})
        if mixing:
            raise NotImplementedError("Sample and mix here, but i'm lazy copying the codes.")
        else:
            ts_style[:, 1] = ts_style[:, 0] # Forcingly and explicitly disable style fusion

        coord_disp = (
            0,  cur_anchor_idx * (self.config.train_params.ts_input_size + self.config.task.anchor_gap)
        )
        coord_shape = (
            self.config.train_params.ts_input_size + self.ss_unfold_pad*2,
            self.config.train_params.ts_input_size*2 + self.config.task.anchor_gap + self.ss_unfold_pad*2,
        )
        local_latent = self.latent_sampler.sample_local_latent(
            self.config.task.batch_size, 
            device="cpu", # Store in CPU anyway, it can be INFINITLY LARGE!
            specific_shape=self.local_latent_shape_single)
        meta_coords = self.coord_handler.sample_coord_grid(
            spatial_latent=None,
            specific_shape=coord_shape, 
            is_training=False,
            coord_init=coord_disp,
            batch_size=self.config.task.batch_size,
            device=self.device)

        # Create randomized noises
        randomized_noises = [
            torch.randn(self.config.task.batch_size, 1, int(h), int(w))
                for (h,w) in zip(self.noise_heights_single, self.noise_widths_single)]

        testing_vars = TestingVars(
            meta_img=meta_img, 
            global_latent=global_latent, 
            local_latent=local_latent, 
            meta_coords=meta_coords, 
            styles=ts_style, 
            noises=randomized_noises, 
            device=self.device)

        return testing_vars


    def fuse_anchors(self, anchor_stack):
        num_centers = len(self.config.task.style_centers)
        left_anchor = anchor_stack[num_centers//2-1]
        right_anchor = anchor_stack[num_centers//2]

        joint_local_latent = self.connect_local_latents(left_anchor.local_latent, right_anchor.local_latent)
        noise_gaps = [torch.randn(self.config.task.batch_size, 1, h, wgap) for h,wgap in zip(self.noise_heights_single, self.noise_gap_sizes)]
        joint_noises = [torch.cat([l,g,r], 3) for l,g,r in zip(left_anchor.noises, noise_gaps, right_anchor.noises)]
        return TestingVars(
            meta_img=left_anchor.meta_img,
            global_latent=[anchor.global_latent for anchor in anchor_stack],
            local_latent=joint_local_latent,
            meta_coords=left_anchor.meta_coords,
            styles=[anchor.styles for anchor in anchor_stack],
            noises=joint_noises,
            device=self.device
        )

    def generate(self, testing_vars, tkinter_pbar=None, update_by_ss_map=None, update_by_ts_map=None, write_gpu_time=False, calc_flops=False):

        cur_fusion_map_ts = list(self.meta_fusion_map_ts)
        cur_fusion_map_ts.append(self.meta_fusion_map_ts[-1])

        # Handle the spatially-shaped global_latents in SS
        cur_fusion_map_ss = self.meta_fusion_map_ss.to(self.device)
        
        # Handle other variables
        cur_local_latent = testing_vars.local_latent.to(self.device)
        cur_coords = testing_vars.meta_coords.to(self.device)
        cur_noises = [n.to(self.device) for n in testing_vars.noises]

        # Hard to define global/style mixing while fusion, viable, but I'm lazy
        single_global_latents = [
            v[:, 0] for v in testing_vars.global_latent]
        
        single_styles = [s[:, 0] for s in testing_vars.styles]
        g_ema_kwargs = {
            "global_latent": single_global_latents,
            "local_latent": cur_local_latent,
            "styles": single_styles,
            "override_coords": cur_coords,
            "noises": cur_noises,
            "disable_dual_latents": True,
            "style_fusion_map_ss": cur_fusion_map_ss,
            "style_fusion_map_ts": cur_fusion_map_ts,
            "calc_flops": calc_flops,
        }
        output = self.g_ema(**g_ema_kwargs)
        testing_vars.meta_img = output["gen"].cpu().detach()

        # Somehow the cache is not released
        del output
        torch.cuda.empty_cache()

    def save_meta_imgs(self, meta_img):

        # Save the full image and the low-resolution image (for visualization)
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        assert self.config.task.batch_size == 1
        for i in range(self.config.task.batch_size):
            save_path = os.path.join(self.save_root, str(self.cur_global_id).zfill(6)+".png")
            plt.imsave(save_path, meta_img_np[i])

        self.cur_global_id += self.config.train_params.batch_size

    def exit(self):
        if self.config.task.save_type == "all-in-one":
            meta_img = torch.cat(self.output_stack, 3)
            self.save_meta_imgs(meta_img)

    def _create_start_pts_mesh(self, step_size, num_steps_h, num_steps_w):
        start_pts_x = np.arange(num_steps_h) * step_size
        start_pts_y = np.arange(num_steps_w) * step_size
        start_pts_mesh = np.stack([
            np.repeat(start_pts_x.reshape(num_steps_h, 1), num_steps_w, axis=1),
            np.repeat(start_pts_y.reshape(1, num_steps_w), num_steps_h, axis=0),
        ], 2).astype(np.uint32) # shape: (H, W, 2)
        return start_pts_mesh

    def _create_idx_grid(self, num, h, w):
        x_idx = torch.arange(h).reshape(h, 1).repeat(1, w) # (h, w)
        y_idx = torch.arange(w).reshape(1, w).repeat(h, 1) # (h, w)
        grid = torch.stack([x_idx, y_idx])
        gird = grid.unsqueeze(0).repeat(num, 1, 1, 1) # (num, 2, h, w)
        return gird

    def _create_fusion_map(self, style_center_list):
        """
        The generator involves cropping to deal with zero-paddings.
        As the spatially-shaped styles will face resizing 
        """
        # input: (h_i, w_i)
        # style: (h_i, w_i)
        # noise: (h_(i+1), w_(i+1))
        # output = conv_trans(input, style) + noise

        fusion_map_pad = self.config.task.fusion_map_pad
        padded_height = self.target_height + fusion_map_pad[0] * 2
        padded_width  = self.target_width + fusion_map_pad[1] * 2

        num_centers = len(style_center_list)
        meta_dist_to_center = self._create_idx_grid(
            num_centers, padded_height, padded_width)
        meta_dist_to_center[:, 0, :, :] = meta_dist_to_center[:, 0, :, :] - fusion_map_pad[0]
        meta_dist_to_center[:, 1, :, :] = meta_dist_to_center[:, 1, :, :] - fusion_map_pad[1]
        for i,style_center in enumerate(style_center_list):
            meta_dist_to_center[i, 0] = meta_dist_to_center[i, 0] - style_center[0]
            meta_dist_to_center[i, 1] = meta_dist_to_center[i, 1] - style_center[1]

        # Use simple L2 dist
        meta_dist_to_center = meta_dist_to_center.pow(2).sum(1) # output shape: (N, H, W), sqrt omitted
        meta_fusion_idx = meta_dist_to_center.topk(1, largest=False, dim=0).indices[0] # (H, W)

        # Convert (H, W) index to (N, H, W) mask, for later blur filtering
        batch_size = self.config.task.batch_size
        meta_fusion_map = torch.zeros(num_centers, padded_height*padded_width) # (N, H*W)
        meta_fusion_idx = meta_fusion_idx.view(1, padded_height*padded_width) # (1, H*W)
        meta_fusion_map.scatter_(0, meta_fusion_idx, 1)
        meta_fusion_map = meta_fusion_map.unsqueeze(0).repeat(batch_size, 1, 1) # (B, N, H*W)
        meta_fusion_map = meta_fusion_map.reshape(batch_size, num_centers, padded_height, padded_width)

        meta_fusion_map = self.blur_kernel(meta_fusion_map)

        # Save initial fusion map for future visualization
        if self.save_root is not None:
            for c in range(num_centers):
                plt.imsave(
                    os.path.join(self.save_root, "_init_fusion_map_{}.png".format(c)),
                    meta_fusion_map[0, c].cpu().detach().clamp(0, 1).numpy())


        # Need to carefully account the upsampling and no-padding involved in the generator
        # [Note] All the latents returned are on the input-side
        meta_fusion_map_ss, meta_fusion_map_ts, _, _ = \
            self.g_ema_module.calibrate_spatial_shape(
                meta_fusion_map, direction="backward", padding_mode="replicate", verbose=False)

        # Remove fusion_map_pad, here we just want to get the shape of intermediate feature maps in each layer
        dummy_tensor = torch.zeros(1, 1, self.target_height, self.target_width)
        dummy_fusion_map_ss, dummy_fusion_map_ts, _, _ = \
            self.g_ema_module.calibrate_spatial_shape(
                dummy_tensor, direction="backward", padding_mode="replicate", verbose=True)
        meta_fusion_map_ss = [
            self._center_crop(src=feat_meta, tgt=feat_dummy) 
                for feat_meta, feat_dummy in zip(meta_fusion_map_ss, dummy_fusion_map_ss)]
        meta_fusion_map_ts = [
            self._center_crop(src=feat_meta, tgt=feat_dummy) 
                for feat_meta, feat_dummy in zip(meta_fusion_map_ts, dummy_fusion_map_ts)]

        # Special case handling for the last ToRgb layer, which is in pixel-space shape
        meta_fusion_map_ts.append(meta_fusion_map)

        # We only use the first (as well as the largest) meta_fusion_map_ss
        meta_fusion_map_ss = meta_fusion_map_ss[0]

        return meta_fusion_map_ss, meta_fusion_map_ts

    def _center_crop(self, src, tgt):
        _, _, sh, sw = src.shape
        _, _, th, tw = tgt.shape
        assert sh>=th and sw>=tw, \
            "src should always >= to tgt shape, got src {} and tgt {}".format(src.shape, tgt.shape)
        pad_h = (sh - th) // 2
        pad_w = (sw - tw) // 2
        return src[:, :, pad_h:pad_h+th, pad_w:pad_w+tw]
    
    def connect_local_latents(self, local_latent_prev, local_latent_next):
        # |    prev   | gap |   next   |
        #                   |   prev   | gap |   next   |
        #                                    |   prev   | gap |   next   |
        #
        #            ss_pad |  anchor  | ss_pad
        # |->   ss_disp   <-|->   ss_disp   <-|->   ss_disp   <-|->   ss_disp   <-|

        # [EX] suppose we have ss_pad=4:
        #
        # anchor_gap=0
        #   prev: ... A B C D|E F G H
        #   next:     A B C D|E F G H ...
        # anchor_gap=1
        #   prev: ... A B C D|E F G H
        #   next:       B C D E|F G H ...
        # anchor_gap=2
        #   prev: ... A B C D|E F G H
        #   next:         C D E F|G H ...
        #
        # (...)
        #
        # anchor_gap=8
        #   prev: ... A B C D|E F G H
        #   next:                     I J K L|M N O P ...
        # anchor_gap=9
        #   prev: ... A B C D|E F G H
        #   new :                     X
        #   next:                       J K L M|N O P ...

        ss_pad = self.ss_unfold_pad
        ts_input_size = self.config.train_params.ts_input_size

        assert local_latent_prev.shape[3] == ss_pad*2 + ts_input_size, \
            "Got {} != {}*2+{} (={})".format(local_latent_prev.shape[3], ss_pad, ts_input_size, ss_pad*2+ts_input_size)

        overlap_size = ss_pad*2 - self.config.task.anchor_gap
        if overlap_size == 0: # ss_pad accidentally equals to gap, no new values to fill in, no need to overwtite values as well
            prev_slice = local_latent_prev
            gap_slice = None
            next_slice = local_latent_next
        elif overlap_size < 0: # ss_pad does not cover the gap, add new local_latent in
            B, C, _, _ = local_latent_prev.shape
            prev_slice = local_latent_prev
            gap_slice = torch.randn(B, C, ts_input_size, -overlap_size).to(self.device)
            next_slice = local_latent_next
        else: # ss_pad overlaps
            prev_slice = local_latent_prev
            gap_slice = None
            next_slice = local_latent_next[:, :, :, overlap_size:]

        if gap_slice is None:
            return torch.cat([prev_slice, next_slice], 3)
        else:
            return torch.cat([prev_slice, gap_slice, next_slice], 3)

    def unify_overlapped_local_latent(self, local_latent_prev, local_latent_next):
        ss_pad = self.ss_unfold_pad
        overlap_size = ss_pad*2 - self.config.task.anchor_gap
        if overlap_size > 0:
            prev_overlap = local_latent_prev[:, :, :, -overlap_size:]
            local_latent_next[:, :, :, :overlap_size] = prev_overlap

    # def dispute_local_latents(self, local_latent):
    #     ss_disp = self.config.train_params.ts_input_size + self.config.task.anchor_gap
    #     assert ss_disp > 0, "Got unexpected displacement {}".format(ss_disp)
    #     return local_latent[:, :, :, ss_disp:]
    
    # def verify_overlapped_anchors(self, anchor_prev, anchor_next):
    #     ss_pad = self.config.train_params.ss_n_layers * self.config.train_params.ss_unfold_radius
    #     ts_input_size = self.config.train_params.ts_input_size
    #     assert anchor_prev.shape[3] == ss_pad*2 + ts_input_size, \
    #         "Got {} != {}*2+{} (={})".format(anchor_prev.shape[3], ss_pad, ts_input_size, ss_pad*2+ts_input_size)

    #     overlap_slice = anchor_prev[:, :, :, -ss_pad:]
    #     assert ((anchor_next[:, :, :, :ss_pad] - overlap_slice).abs() < 1e-2).all(), "Got overlapped achors have in consistent values!"
 

    def compose_inv_root(self):
        if isinstance(self.config.task.prev_inv_config, list):
            return [
                os.path.join("./logs/", self.config.var.exp_name, "test", cfg, "stats")
                    for cfg in self.config.task.prev_inv_config]
        else:
            return [os.path.join("./logs/", self.config.var.exp_name, "test", self.config.task.prev_inv_config, "stats")]
 
