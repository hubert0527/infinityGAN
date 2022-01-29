import os
import math
import datetime
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from itertools import product as iter_product
from skimage.transform import resize as sk_resize

import torch

from utils import manually_seed
from test_managers.base_test_manager import BaseTestManager
from test_managers.testing_vars_wrapper import TestingVars


class StructureTextureTraitMaker(BaseTestManager):

    def task_specific_init(self, output_size=None):

        assert self.config.train_params.batch_size == 1, "Made in a hurry, not considered."

        if hasattr(self.config.task, "pano_mode") and self.config.task.pano_mode:
            raise NotImplementedError()

        if output_size is None:
            self.target_height = self.config.task.height
            self.target_width = self.config.task.width
        else:
            self.target_height, self.target_width = output_size
           
        self._init_starting_points()         
        self.noise_heights = self.outfeat_step_sizes * (self.num_steps_h-1) + self.outfeat_sizes_list
        self.noise_widths  = self.outfeat_step_sizes * (self.num_steps_w-1) + self.outfeat_sizes_list


    def run_next(self, save=True, write_gpu_time=False, inv_records=None, inv_placements=None, calc_flops=False, **kwargs):
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if v is not None:
                    print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))

        # Only create once for trait generation
        if self.cur_global_id == 0:
            testing_vars = self.create_vars(inv_records=inv_records, inv_placements=inv_placements)
            self.tmp_testing_vars = testing_vars
        else:
            testing_vars = self.tmp_testing_vars

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
        #     128GB CPU memory OuO
        # 2.  Historically, we call non-processed spatial_latent as local_latent.
        full_local_latent_shape = (
            # Does not account GNN padding here, it is handled within the latent_sampler
            int(self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)),
            int(self.g_ema_module.calc_in_spatial_size(self.meta_width, include_ss=False)),
        )

        global_sample_fn = \
            lambda: self.latent_sampler.sample_global_latent(
                self.config.train_params.batch_size, mixing=mixing, device=self.device)
        local_sample_fn = \
            lambda: self.latent_sampler.sample_local_latent(
                self.config.train_params.batch_size, 
                device="cpu", # Store in CPU anyway, it can be INFINITLY LARGE!
                specific_shape=full_local_latent_shape)
        def style_sample_fn():
            tmp_global_latent = self.latent_sampler.sample_global_latent(
                self.config.train_params.batch_size, mixing=mixing, device=self.device)
            tmp_styles = self.g_ema(
                call_internal_method="get_style", 
                internal_method_kwargs={"global_latent": tmp_global_latent})
            if mixing:
                raise NotImplementedError("Sample and mix here, but i'm lazy copying the codes.")
            else:
                n_latent = self.g_ema_module.texture_synthesizer.n_latent
                tmp_styles = tmp_styles[:, 0:1].repeat(1, n_latent, 1) # Forcingly and explicitly disable style fusion
            return tmp_styles

        sampling_fns = {
            "global_latent": global_sample_fn,
            "local_latent": local_sample_fn,
            "styles": style_sample_fn,
        }

        # Create randomized noises
        randomized_noises = [
            torch.randn(self.config.train_params.batch_size, 1, int(h), int(w))
                for (h,w) in zip(self.noise_heights, self.noise_widths)]

        meta_coords = self.coord_handler.sample_coord_grid(
            local_sample_fn(), 
            is_training=False)

        testing_vars = TestingVars(
            meta_img=meta_img, 
            global_latent=global_sample_fn(), 
            styles=None,
            local_latent=local_sample_fn(), 
            meta_coords=meta_coords, 
            noises=randomized_noises, 
            device=self.device)

        # Setup trait data
        n_trait_x = self.config.task.n_trait_x
        n_trait_y = self.config.task.n_trait_y

        setattr(
            testing_vars, 
            self.config.task.trait_x, 
            [sampling_fns[self.config.task.trait_x]() for _ in range(n_trait_x)])
        setattr(
            testing_vars, 
            self.config.task.trait_y, 
            [sampling_fns[self.config.task.trait_y]() for _ in range(n_trait_y)])

        for i,s in zip(self.config.task.random_reset_x, self.config.task.reset_seed_x):
            manually_seed(s)
            getattr(testing_vars, self.config.task.trait_x)[i] = sampling_fns[self.config.task.trait_x]()

        for i,s in zip(self.config.task.random_reset_y, self.config.task.reset_seed_y):
            manually_seed(s)
            getattr(testing_vars, self.config.task.trait_y)[i] = sampling_fns[self.config.task.trait_y]()


        if inv_records is not None:
            raise NotImplementedError()
            # testing_vars.replace_by_records(
            #     self.g_ema_module, inv_records, inv_placements, assert_no_style=True)
        
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
            # Trait generation usually short
            pbar = idx_tuples

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

            n_trait_x = self.config.task.n_trait_x
            n_trait_y = self.config.task.n_trait_y
            cur_trait_x_pos = self.cur_global_id // n_trait_y
            cur_trait_y_pos = self.cur_global_id % n_trait_y

            g_ema_kwargs = {
                "global_latent": testing_vars.global_latent,
                "styles": None,
                "local_latent": testing_vars.local_latent,
                "override_coords": testing_vars.meta_coords,
                "noises": noises,
                "disable_dual_latents": True,
                "calc_flops": calc_flops,
            }

            g_ema_kwargs[self.config.task.trait_x] = getattr(testing_vars, self.config.task.trait_x)[cur_trait_x_pos]
            g_ema_kwargs[self.config.task.trait_y] = getattr(testing_vars, self.config.task.trait_y)[cur_trait_y_pos]

            g_ema_kwargs["local_latent"] = g_ema_kwargs["local_latent"][:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            g_ema_kwargs["override_coords"] = g_ema_kwargs["override_coords"][:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            
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

            # cur_local_latent = testing_vars.local_latent[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            # cur_coords = testing_vars.meta_coords[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            # output = self.g_ema(
            #     global_latent=testing_vars.global_latent, 
            #     local_latent=cur_local_latent,
            #     override_coords=cur_coords,
            #     noises=noises,
            #     disable_dual_latents=True)

            # # Historical issue...
            # patch = output["gen"] if "gen" in output else output["patch"]

            # img_x_st, img_y_st = outfeat_x_st[-1], outfeat_y_st[-1]
            # img_x_ed, img_y_ed = outfeat_x_ed[-1], outfeat_y_ed[-1]
            # testing_vars.meta_img[:, :, img_x_st:img_x_ed, img_y_st:img_y_ed] = patch.detach().cpu()

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

        n_trait_samples = self.config.task.n_trait_x * self.config.task.n_trait_y
        if self.cur_global_id == n_trait_samples:
            self.compose_trait_imgs()

    def save_testing_vars(self, testing_vars):
        assert self.config.train_params.batch_size == 1, \
            "This is only designed to be used with the interactive tool."
        save_path = os.path.join(self.save_root, str(self.cur_global_id).zfill(6)+".pkl")
        pkl.dump(testing_vars, open(save_path, "wb"))


    def compose_trait_imgs(self, pad=4):
        prev_img_paths = sorted(glob(os.path.join(self.save_root, "*.png")))
        imgs = [plt.imread(p) for p in prev_img_paths]

        n_trait_x = self.config.task.n_trait_x
        n_trait_y = self.config.task.n_trait_y
        trait_size_h = (self.config.task.height * n_trait_x) + (n_trait_x-1) * pad
        trait_size_w = (self.config.task.width * n_trait_y) + (n_trait_y-1) * pad
        trait_img = np.ones((trait_size_h, trait_size_w, 3), dtype=imgs[0].dtype)
        for i in range(n_trait_x):
            for j in range(n_trait_y):
                sid = i * n_trait_y + j
                xst = i * self.config.task.height + i * pad
                yst = j * self.config.task.width + j * pad
                xed = xst + self.config.task.height
                yed = yst + self.config.task.width
                trait_img[xst:xed, yst:yed, :] = imgs[sid][:, :, :3]

        output_paths = os.path.join(self.save_root, "diversity_trait.png")
        plt.imsave(output_paths, trait_img)


    def save_meta_imgs(self, meta_img, rotate=0):

        # Center crop
        pad_h = (self.meta_height - self.target_height) // 2
        pad_w = (self.meta_width - self.target_width) // 2
        meta_img = meta_img[:, :, pad_h:pad_h+self.target_height, pad_w:pad_w+self.target_width]

        if abs(rotate) > 1e-6:
            n_pix = int(round(meta_img.shape[3] * rotate))
            meta_img = torch.cat([
                meta_img[:, :, :, n_pix:],
                meta_img[:, :, :, :n_pix],
            ], 3)

        # Save the full image and the low-resolution image (for visualization)
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i
            if abs(rotate) > 1e-6:
                rotate_suffix = "_rot{:.2f}".format(rotate)
            else:
                rotate_suffix = ""

            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+rotate_suffix+".png")
            plt.imsave(save_path, meta_img_np[i])
    
            # lr_save_path = os.path.join(self.save_root, str(global_id).zfill(6)+rotate_suffix+"_lr.png")
            # resize_ratio = self.config.task.lowres_height / self.config.task.height
            # resize_shape  = (
            #     int(round(self.target_height*resize_ratio)), 
            #     int(round(self.target_width*resize_ratio)))
            # lr_img = sk_resize((meta_img_np[i]*255).astype(np.uint8), resize_shape)
            # plt.imsave(lr_save_path, lr_img)


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
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + 1
        self.num_steps_w = \
            math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + 1

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
 
