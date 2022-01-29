import os
import math
import datetime
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from itertools import product as iter_product

import torch

from test_managers.base_test_manager import BaseTestManager
from test_managers.testing_vars_wrapper import TestingVars
from test_managers.global_config import test_meta_extra_pad


class InfiniteGenerationManager(BaseTestManager):

    def task_specific_init(self, output_size=None):

        if output_size is None:
            self.target_height = self.config.task.height
            self.target_width = self.config.task.width
        else:
            self.target_height, self.target_width = output_size
           
        self._init_starting_points()         
        self.noise_heights = self.outfeat_step_sizes * (self.num_steps_h-1) + self.outfeat_sizes_list
        self.noise_widths  = self.outfeat_step_sizes * (self.num_steps_w-1) + self.outfeat_sizes_list

        if hasattr(self.config.task, "gen_from_inv_stats") and self.config.task.gen_from_inv_stats:
            self.inv_root = self.compose_inv_root()
            self.inv_rec_files = sorted(glob(os.path.join(self.inv_root, "*")))
            self.gen_from_inv_stats = True
        else:
            self.gen_from_inv_stats = False

    def run_next(self, save=True, write_gpu_time=False, inv_records=None, inv_placements=None, calc_flops=False, disable_pbar=False, **kwargs):
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if v is not None:
                    print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))
        testing_vars = self.create_vars(inv_records=inv_records, inv_placements=inv_placements)
        self.generate(testing_vars, write_gpu_time=write_gpu_time, calc_flops=calc_flops, disable_pbar=disable_pbar)
        if save:
            self.save_results(testing_vars.meta_img)
        return testing_vars.meta_img

    def create_vars(self, inv_records=None, inv_placements=None):
        mixing = False
        assert mixing == False, "Otherwise, an injection index must be specified and fed into g_ema."

        # Allocate memory for the final output, starts iterating and filling in the generated results.
        # Can be reused
        meta_img = torch.empty(
            self.config.train_params.batch_size,
            3,
            int(self.meta_height),
            int(self.meta_width)).float()

        # [Note]
        # 1.  One may implement a sophisticated version that does not required to 
        #     generate all the latents at once, as most of the info are not reusing
        #     during the inference. However, the author is just lazy and abusing his 
        #     CPU memory OuO
        global_latent = self.latent_sampler.sample_global_latent(
            self.config.train_params.batch_size, mixing=mixing, device=self.device)
        full_local_latent_shape = (
            # Does not account GNN padding here, it is handled within the latent_sampler
            int(self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)),
            int(self.g_ema_module.calc_in_spatial_size(self.meta_width, include_ss=False)),
        )
        local_latent = self.latent_sampler.sample_local_latent(
            self.config.train_params.batch_size, 
            device="cpu", # Store in CPU anyway, it can be INFINITLY LARGE!
            specific_shape=full_local_latent_shape)

        meta_coords = self.coord_handler.sample_coord_grid(
            local_latent, 
            is_training=False)

        # Create randomized noises
        randomized_noises = [
            torch.randn(self.config.train_params.batch_size, 1, int(h), int(w))
                for (h,w) in zip(self.noise_heights, self.noise_widths)]

        testing_vars = TestingVars(
            meta_img=meta_img, 
            global_latent=global_latent, 
            local_latent=local_latent, 
            meta_coords=meta_coords, 
            noises=randomized_noises, 
            device=self.device)

        if self.gen_from_inv_stats:
            assert inv_records is None, \
                "`gen_from_inv_stats` already specified, should not receive `inv_records` from command!"
            assert self.config.train_params.batch_size == 1, \
                "Inverted parameters loading for batch is not yet implemeted! " + \
                "Please use parallel-batching instead, which provides a similar inference speed."
            inv_records = [self.inv_rec_files[self.cur_global_id]]
            inv_placements = [self.config.task.gen_from_inv_placement]

        if inv_records is not None:
            testing_vars.replace_by_records(
                self.g_ema_module, inv_records, inv_placements, assert_no_style=True)
        
        return testing_vars

    def generate(self, testing_vars, tkinter_pbar=None, update_by_ss_map=None, update_by_ts_map=None, 
                 write_gpu_time=False, calc_flops=False, disable_pbar=False):

        # I don't mind bruteforce casting combination here, cuz you should worry about the meta_img size first
        idx_tuples = list(iter_product(range(self.start_pts_mesh_z.shape[0]), range(self.start_pts_mesh_z.shape[1])))

        if disable_pbar:
            pbar = idx_tuples
        elif tkinter_pbar is not None:
            pbar = tkinter_pbar(idx_tuples)
        else:
            pbar = tqdm(idx_tuples)

        accum_exec_time = 0
        accum_flops_all, accum_flops_ss, accum_flops_ts = 0, 0, 0
        for iiter, (idx_x,idx_y) in enumerate(pbar):
            zx_st, zy_st = self.start_pts_mesh_z[idx_x, idx_y]
            zx_ed = zx_st + self.config.train_params.ts_input_size 
            zy_ed = zy_st + self.config.train_params.ts_input_size

            # Handle the randomized noise input of the texture_synthesizer...
            outfeat_x_st = [start_pts_mesh[idx_x,idx_y,0] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_y_st = [start_pts_mesh[idx_x,idx_y,1] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_x_ed = [
                x_st + out_size for (x_st, out_size) in zip(outfeat_x_st, self.outfeat_sizes_list)]
            outfeat_y_ed = [
                y_st + out_size for (y_st, out_size) in zip(outfeat_y_st, self.outfeat_sizes_list)]
            noises = []
            for i, (fx_st, fy_st, fx_ed, fy_ed) in enumerate(zip(outfeat_x_st, outfeat_y_st, outfeat_x_ed, outfeat_y_ed)):
                noises.append(testing_vars.noises[i][:, :, fx_st:fx_ed, fy_st:fy_ed].to(self.device))

            # Deal with SS unfolding here
            zx_st -= self.ss_unfold_size
            zy_st -= self.ss_unfold_size
            zx_ed += self.ss_unfold_size
            zy_ed += self.ss_unfold_size
            
            # [Interactive] Decide whether the region will be updated, otherwise no need to generate
            if update_by_ss_map is not None:
                ss_cursors = zx_st, zx_ed, zy_st, zy_ed
                if not self.is_overlaping_update_map(update_by_ss_map, *ss_cursors):
                    continue
            if update_by_ts_map is not None:
                # For TS regional selection, we only select noises
                ts_cursors = outfeat_x_st[0], outfeat_x_ed[0], outfeat_y_st[0], outfeat_y_ed[0]
                if not self.is_overlaping_update_map(update_by_ts_map, *ts_cursors):
                    continue

            cur_local_latent = testing_vars.local_latent[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            cur_coords = testing_vars.meta_coords[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            g_ema_kwargs = {
                "global_latent": testing_vars.global_latent,
                "local_latent": cur_local_latent,
                "override_coords": cur_coords,
                "noises": noises,
                "disable_dual_latents": True,
                "calc_flops": calc_flops,
            }
            if hasattr(testing_vars, "wplus_styles") and testing_vars.wplus_styles is not None:
                g_ema_kwargs["wplus_styles"] = testing_vars.wplus_styles
            img_x_st, img_y_st = outfeat_x_st[-1], outfeat_y_st[-1]
            img_x_ed, img_y_ed = outfeat_x_ed[-1], outfeat_y_ed[-1]
            index_tuple = (img_x_st, img_x_ed, img_y_st, img_y_ed)

            exec_time, flops = self.maybe_parallel_inference(
                testing_vars, g_ema_kwargs=g_ema_kwargs, index_tuple=index_tuple, return_exec_time=write_gpu_time, calc_flops=calc_flops)
            accum_exec_time += exec_time
            if calc_flops:
                accum_flops_all += flops["all"]
                accum_flops_ss += flops["ss"]
                accum_flops_ts += flops["ts"]

        exec_time, flops = self.maybe_parallel_inference(
            testing_vars, flush=True, return_exec_time=write_gpu_time, calc_flops=calc_flops)
        if calc_flops:
            accum_flops_all += flops["all"]
            accum_flops_ss += flops["ss"]
            accum_flops_ts += flops["ts"]
        
        if write_gpu_time:
            accum_exec_time += exec_time
            print(" [*] GPU time {}".format(accum_exec_time))
            self.accum_exec_times.append(accum_exec_time)
            fmt_date = datetime.date.today().strftime("%d-%m-%Y")
            benchmark_file = os.path.join(self.save_root, "speed_benchmark_{}.txt".format(fmt_date))
            with open(benchmark_file, "a") as f:
                f.write("{:.6f}".format(accum_exec_time))

        if calc_flops:
            print(" [*] Total FLOPs: {} (SS {}, TS {})".format(
                self.pretty_print_flops(accum_flops_all), 
                self.pretty_print_flops(accum_flops_ss), 
                self.pretty_print_flops(accum_flops_ts)))


    def save_results(self, meta_img, dump_vars=None):
        print(" [*] Saving results...")
        self.save_meta_imgs(meta_img)
        if dump_vars is not None:
            self.save_testing_vars(dump_vars)
        self.cur_global_id += self.config.train_params.batch_size

    def save_testing_vars(self, testing_vars):
        assert self.config.train_params.batch_size == 1, \
            "This is only designed to be used with the interactive tool."
        save_path = os.path.join(self.save_root, str(self.cur_global_id).zfill(6)+".pkl")
        pkl.dump(testing_vars, open(save_path, "wb"))

    def _wrap_feature(self, feat, wrap_size, dim):
        assert wrap_size < (feat.shape[dim] - 2*wrap_size), \
            "Does not expect the wrapping area is larger than a full period."
        if dim == 2:
            valid_st = feat[:, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, -2*wrap_size:-wrap_size]
            feat[:, :, :wrap_size] = valid_ed
            feat[:, :, -wrap_size:] = valid_st
        elif dim == 3:
            valid_st = feat[:, :, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, :, -2*wrap_size:-wrap_size]
            feat[:, :, :, :wrap_size] = valid_ed
            feat[:, :, :, -wrap_size:] = valid_st
        else:
            raise NotImplementedError(
                "I don't expect this function will be used other than spatial dims, but got {}.".format(dim))
        
    def save_meta_imgs(self, meta_img):

        # Center crop
        pad_h = (self.meta_height - self.target_height) // 2
        pad_w = (self.meta_width - self.target_width) // 2
        meta_img = meta_img[:, :, pad_h:pad_h+self.target_height, pad_w:pad_w+self.target_width]

        # Save the full image and the low-resolution image (for visualization)
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+".png")
            plt.imsave(save_path, meta_img_np[i])

    def _create_start_pts_mesh(self, step_size, num_steps_h, num_steps_w):
        start_pts_x = np.arange(num_steps_h) * step_size
        start_pts_y = np.arange(num_steps_w) * step_size
        start_pts_mesh = np.stack([
            np.repeat(start_pts_x.reshape(num_steps_h, 1), num_steps_w, axis=1),
            np.repeat(start_pts_y.reshape(1, num_steps_w), num_steps_h, axis=0),
        ], 2).astype(np.uint32) # shape: (H, W, 2)
        return start_pts_mesh


    def _init_starting_points(self):

        # Get all the starting points of the patches to generate
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        self.num_steps_w = \
            math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad

        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.pixelspace_step_size * (self.num_steps_w-1) + self.outfeat_sizes_list[-1]
 

    def compose_inv_root(self):
        return os.path.join("./logs/", self.config.var.exp_name, "test", self.config.task.prev_inv_config, "stats")
        
