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

from models.ops import Blur
from test_managers.base_test_manager import BaseTestManager
from test_managers.testing_vars_wrapper import TestingVars
from test_managers.global_config import test_meta_extra_pad


class FusedGenerationManager(BaseTestManager):

    def task_specific_init(self):

        self.target_height = self.config.task.height
        self.target_width = self.config.task.width
        self.ss_unfold_size = self.config.train_params.ss_n_layers * self.config.train_params.ss_unfold_radius

        self._init_starting_points()
        self.noise_heights = self.outfeat_step_sizes * (self.num_steps_h-1) + self.outfeat_sizes_list
        self.noise_widths  = self.outfeat_step_sizes * (self.num_steps_w-1) + self.outfeat_sizes_list

        self.meta_local_latent_shape = (
            # Does not account GNN padding here, it is handled within the latent_sampler
            self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False),
            self.g_ema_module.calc_in_spatial_size(self.meta_width, include_ss=False),
        )

        # The meta size is different from the direct output size
        self.x_meta_pad_h = (self.meta_height - self.target_height) // 2
        self.x_meta_pad_w = (self.meta_width  - self.target_width) // 2
        self.meta_style_center_indices = [
            [
                center_f[0] * self.target_height + self.x_meta_pad_h,
                center_f[1] * self.target_width  + self.x_meta_pad_w,
            ] for center_f in self.config.task.style_centers]

        self.blur_kernel = Blur(
            kernel=self.config.task.blur_kernel,
            pad=self.config.task.blur_kernel//2,
            padding_mode="replicate",
            prior=self.config.task.blur_type)

        # This is used to select which style to use at each spatial index
        print(" [*] Creating fusion map...")
        self.meta_fusion_map_ss, self.meta_fusion_map_ts = self._create_fusion_map(
            self.meta_style_center_indices)

        if hasattr(self.config.task, "gen_from_inv_stats") and self.config.task.gen_from_inv_stats:
            self.inv_roots = self.compose_inv_root()
            self.inv_rec_files = [sorted(glob(os.path.join(root, "*"))) for root in self.inv_roots]
            self.gen_from_inv_stats = True
        else:
            self.gen_from_inv_stats = False

    def run_next(self, save=True, write_gpu_time=False, inv_records=None, inv_placements=None, calc_flops=False, disable_pbar=True, **kwargs):
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if v is not None:
                    print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))
        testing_vars = self.create_vars(inv_records=inv_records, inv_placements=inv_placements)
        self.generate(testing_vars, write_gpu_time=write_gpu_time, calc_flops=calc_flops)
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
        all_global_latents = []
        all_styles = []
        for i in range(len(self.config.task.style_centers)):
            # [Hubert] Actually, can use mixing here, but disabled by default for simplicity
            global_latent = self.latent_sampler.sample_global_latent(
                self.config.train_params.batch_size, mixing=mixing, device=self.device)
            # cur_ts_style = self.g_ema_module.texture_synthesizer.get_style(global_latent)
            cur_ts_style = self.g_ema(
                call_internal_method="get_style", 
                internal_method_kwargs={"global_latent": global_latent})
            if mixing:
                raise NotImplementedError("Sample and mix here, but i'm lazy copying the codes.")
            else:
                cur_ts_style[:, 1] = cur_ts_style[:, 0] # Forcingly and explicitly disable style fusion
                global_latent = global_latent
            all_global_latents.append(global_latent)
            all_styles.append(cur_ts_style)
        local_latent = self.latent_sampler.sample_local_latent(
            self.config.train_params.batch_size, 
            device="cpu", # Store in CPU anyway, it can be INFINITLY LARGE!
            specific_shape=self.meta_local_latent_shape)
        meta_coords = self.coord_handler.sample_coord_grid(
            local_latent, 
            is_training=False)

        # Create randomized noises
        randomized_noises = [
            torch.randn(self.config.train_params.batch_size, 1, int(h), int(w))
                for (h,w) in zip(self.noise_heights, self.noise_widths)]

        testing_vars = TestingVars(
            meta_img=meta_img, 
            global_latent=all_global_latents, 
            local_latent=local_latent, 
            meta_coords=meta_coords, 
            styles=all_styles, 
            noises=randomized_noises, 
            device=self.device)

        if self.gen_from_inv_stats and len(self.inv_roots) > 0:
            assert inv_records is None, \
                "`gen_from_inv_stats` already specified, should not receive `inv_records` from command!"
            assert self.config.train_params.batch_size == 1, \
                "Inverted parameters loading for batch is not yet implemeted! " + \
                "Please use parallel-batching instead, which provides a similar inference speed."
            num_style_centers = len(self.config.task.style_centers)
            num_inv_records = len(self.inv_roots)
            assert num_style_centers >= num_inv_records, "Got {} < {}".format(num_style_centers, num_inv_records)

            inv_records, inv_placements = [], []
            for i in range(num_style_centers):
                if i < num_inv_records:
                    inv_records.append(self.inv_rec_files[i][self.cur_global_id])
                    inv_placements.append(self.config.task.gen_from_inv_placement[i])
                else:
                    inv_records.append(None)
                    inv_placements.append(None)

        if inv_records is not None:
            testing_vars.replace_by_records(self.g_ema_module, inv_records, inv_placements)

        return testing_vars

    def generate(self, testing_vars, tkinter_pbar=None, update_by_ss_map=None, update_by_ts_map=None, write_gpu_time=False, calc_flops=False):

        # I don't mind bruteforce casting combination here, cuz you should worry about the meta_img size first
        idx_tuples = list(iter_product(range(self.start_pts_mesh_z.shape[0]), range(self.start_pts_mesh_z.shape[1])))

        if tkinter_pbar is not None:
            pbar = tkinter_pbar(idx_tuples)
        else:
            pbar = tqdm(idx_tuples)

        accum_exec_time = 0
        accum_flops_all, accum_flops_ss, accum_flops_ts = 0, 0, 0
        for iiter, (idx_x,idx_y) in enumerate(pbar):

            zx_st, zy_st = self.start_pts_mesh_z[idx_x, idx_y]
            zx_ed = zx_st + self.config.train_params.ts_input_size 
            zy_ed = zy_st + self.config.train_params.ts_input_size

            # Deal with SS unfolding here
            zx_st -= self.ss_unfold_size
            zy_st -= self.ss_unfold_size
            zx_ed += self.ss_unfold_size
            zy_ed += self.ss_unfold_size

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

            # Get the pixel-space cursors
            img_x_st, img_y_st = outfeat_x_st[-1], outfeat_y_st[-1]
            img_x_ed, img_y_ed = outfeat_x_ed[-1], outfeat_y_ed[-1]

            # Handle the spatially-shaped styles in TS
            infeat_x_st = [start_pts_mesh[idx_x,idx_y,0] for start_pts_mesh in self.start_pts_mesh_infeats]
            infeat_y_st = [start_pts_mesh[idx_x,idx_y,1] for start_pts_mesh in self.start_pts_mesh_infeats]
            infeat_x_ed = [
                x_st + out_size for (x_st, out_size) in zip(infeat_x_st, self.infeat_sizes_list)]
            infeat_y_ed = [
                y_st + out_size for (y_st, out_size) in zip(infeat_y_st, self.infeat_sizes_list)]
            cur_fusion_map_ts = []
            for i, (fx_st, fy_st, fx_ed, fy_ed) in enumerate(zip(infeat_x_st, infeat_y_st, infeat_x_ed, infeat_y_ed)):
                cur_fusion_map_ts.append(self.meta_fusion_map_ts[i][:, :, fx_st:fx_ed, fy_st:fy_ed])

            # One extra style for the last ToRgb layer, which is in pixel-space shape
            cur_fusion_map_ts.append(self.meta_fusion_map_ts[-1][:, :, img_x_st:img_x_ed, img_y_st:img_y_ed])

            # Handle the spatially-shaped global_latents in SS
            cur_fusion_map_ss = self.meta_fusion_map_ss[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            
            # Handle other variables
            cur_local_latent = testing_vars.local_latent[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            cur_coords = testing_vars.meta_coords[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)

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

            # Hard to define global/style mixing while fusion, viable, but I'm lazy
            single_global_latents = [
                v[:, 0] for v in testing_vars.global_latent]
            
            if hasattr(testing_vars, "wplus_styles"):
                g_ema_kwargs = {
                    "global_latent": single_global_latents,
                    "local_latent": cur_local_latent,
                    "wplus_styles": testing_vars.wplus_styles,
                    "override_coords": cur_coords,
                    "noises": noises,
                    "disable_dual_latents": True,
                    "style_fusion_map_ss": cur_fusion_map_ss,
                    "style_fusion_map_ts": cur_fusion_map_ts,
                    "calc_flops": calc_flops,
                }
            else:
                single_styles = [
                    s[:, 0] for s in testing_vars.styles]
                g_ema_kwargs = {
                    "global_latent": single_global_latents,
                    "local_latent": cur_local_latent,
                    "styles": single_styles,
                    "override_coords": cur_coords,
                    "noises": noises,
                    "disable_dual_latents": True,
                    "style_fusion_map_ss": cur_fusion_map_ss,
                    "style_fusion_map_ts": cur_fusion_map_ts,
                    "calc_flops": calc_flops,
                }
            index_tuple = (img_x_st, img_x_ed, img_y_st, img_y_ed)
            exec_time, flops = self.maybe_parallel_inference(
                testing_vars, g_ema_kwargs=g_ema_kwargs, index_tuple=index_tuple, return_exec_time=write_gpu_time, calc_flops=calc_flops)
            accum_exec_time += exec_time
            if calc_flops:
                accum_flops_all += flops["all"]
                accum_flops_ss += flops["ss"]
                accum_flops_ts += flops["ts"]

        exec_time, flops = self.maybe_parallel_inference(testing_vars, flush=True, return_exec_time=write_gpu_time, calc_flops=calc_flops)
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


    def _create_fusion_map(self, style_center_list):
        """
        The generator involves cropping to deal with zero-paddings.
        As the spatially-shaped styles will face resizing 
        """
        # input: (h_i, w_i)
        # style: (h_i, w_i)
        # noise: (h_(i+1), w_(i+1))
        # output = conv_trans(input, style) + noise

        self.x_meta_pad_h = (self.meta_height - self.target_height) // 2
        self.x_meta_pad_w = (self.meta_width  - self.target_width) // 2

        num_centers = len(style_center_list)
        meta_dist_to_center = self._create_idx_grid(num_centers, self.meta_height, self.meta_width)
        for i,style_center in enumerate(style_center_list):
            meta_dist_to_center[i, 0] = meta_dist_to_center[i, 0] - style_center[0]
            meta_dist_to_center[i, 1] = meta_dist_to_center[i, 1] - style_center[1]

        # Use simple L2 dist
        meta_dist_to_center = meta_dist_to_center.pow(2).sum(1) # output shape: (N, H, W), sqrt omitted
        meta_fusion_idx = meta_dist_to_center.topk(1, largest=False, dim=0).indices[0] # (H, W)

        # Convert (H, W) index to (N, H, W) mask, for later blur filtering
        batch_size = self.config.train_params.batch_size
        meta_fusion_map = torch.zeros(num_centers, self.meta_height*self.meta_width) # (N, H*W)
        meta_fusion_idx = meta_fusion_idx.view(1, self.meta_height*self.meta_width) # (1, H*W)
        meta_fusion_map.scatter_(0, meta_fusion_idx, 1)
        meta_fusion_map = meta_fusion_map.unsqueeze(0).repeat(batch_size, 1, 1) # (B, N, H*W)
        meta_fusion_map = meta_fusion_map.reshape(batch_size, num_centers, self.meta_height, self.meta_width)

        meta_fusion_map = self.blur_kernel(meta_fusion_map)

        # Save initial fusion map for future visualization
        for i in range(num_centers):
            plt.imsave(
                os.path.join(self.save_root, "_init_fusion_map_{}.png".format(i)),
                meta_fusion_map[0, i].cpu().detach().clamp(0, 1).numpy())

        # Need to carefully account the upsampling and no-padding involved in the generator
        # [Note] All the latents returned are on the input-side
        meta_fusion_map_ss, meta_fusion_map_ts, _, _ = \
            self.g_ema_module.calibrate_spatial_shape(
                meta_fusion_map, direction="backward", padding_mode="replicate", verbose=True)

        # Special case handling for the last ToRgb layer, which is in pixel-space shape
        meta_fusion_map_ts.append(meta_fusion_map)

        # We only use the first (as well as the largest) meta_fusion_map_ss
        meta_fusion_map_ss = meta_fusion_map_ss[0]

        return meta_fusion_map_ss, meta_fusion_map_ts

    def _create_idx_grid(self, num, h, w):
        x_idx = torch.arange(h).reshape(h, 1).repeat(1, w) # (h, w)
        y_idx = torch.arange(w).reshape(1, w).repeat(h, 1) # (h, w)
        grid = torch.stack([x_idx, y_idx])
        gird = grid.unsqueeze(0).repeat(num, 1, 1, 1) # (num, 2, h, w)
        return gird

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

        # Consequently, create for input space
        self.start_pts_mesh_infeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.infeat_step_sizes]

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.pixelspace_step_size * (self.num_steps_w-1) + self.outfeat_sizes_list[-1]
 

    def compose_inv_root(self):
        if isinstance(self.config.task.prev_inv_config, list):
            return [
                os.path.join("./logs/", self.config.var.exp_name, "test", cfg, "stats")
                    for cfg in self.config.task.prev_inv_config]
        else:
            return [os.path.join("./logs/", self.config.var.exp_name, "test", self.config.task.prev_inv_config, "stats")]
 
