import os
import time
import torch
import datetime
import traceback
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize as sk_resize

from test_managers.base_test_manager import BaseTestManager
from latent_sampler import LatentSampler


class DirectGenerationManager(BaseTestManager):

    def __init__(self, g_ema, device, save_root, config):
        self.g_ema = g_ema
        self.device = device
        self.save_root = save_root 
        self.config = config
        self.cur_global_id = 0

        if config.var.dataparallel:
            self.g_ema_module = g_ema.module
        else:
            self.g_ema_module = g_ema

        self.accum_exec_times = []
        self.target_height = self.config.task.height
        self.target_width = self.config.task.width

        self.latent_sampler = LatentSampler(g_ema, config)

        # assert config.train_params.styleGAN2_baseline

        assert self.target_height == self.target_width, \
            "Didn't spend time to make it flexible."

        if hasattr(config.train_params, "styleGAN2_baseline") and config.train_params.styleGAN2_baseline:
            mult = int(np.ceil(self.target_height / self.config.train_params.patch_size))
            self.full_ts_input_size = self.config.train_params.ts_input_size * mult
        else:
            self.full_ts_input_size = self.g_ema_module.calc_in_spatial_size(
                    self.target_height, include_ss=False, return_list=False)
        print(" [*] Full local latent input size:", self.full_ts_input_size)

    def task_specific_init(self):
        pass

    def run_next(self, save=True, write_gpu_time=False, calc_flops=False, **kwargs):
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if k in {"inv_records", "inv_placements"}: continue
                print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))
        meta_img = self.generate(write_gpu_time=write_gpu_time, calc_flops=calc_flops)
        if save:
            self.save_results(meta_img)
        return meta_img

    def generate(self, write_gpu_time=False, calc_flops=False):

        accum_exec_time = 0

        assert self.target_height == self.target_width, \
            "Didn't spend time to make it flexible."
        

        global_latent = self.latent_sampler.sample_global_latent(
            self.config.train_params.batch_size, self.device)
        local_latent = self.latent_sampler.sample_local_latent(
            self.config.train_params.batch_size, 
            self.device, 
            specific_shape=(self.full_ts_input_size, self.full_ts_input_size))

        if write_gpu_time:
            torch.cuda.synchronize()
            start_time = time.time()
            
        try:
            output = self.g_ema(
                global_latent=global_latent,
                local_latent=local_latent,
                disable_dual_latents=True,
                calc_flops=calc_flops).detach().cpu()
            meta_img = output["gen"]
        except RuntimeError as e: # Out of memory
            if "CUDA out of memory" in traceback.format_exc():
                exec_time = -1
                meta_img = None
                output = None
                print(" [!] OOM!")
            else:
                raise e

        if meta_img is not None and write_gpu_time:
            torch.cuda.synchronize()
            exec_time = time.time() - start_time
            
        if calc_flops and (output is not None):
            if "flops_ss" in output:
                print(" [*] Total FLOPs: {} (SS {}, TS {})".format(
                    self.pretty_print_flops(output["flops_all"]), 
                    self.pretty_print_flops(output["flops_ss"]), 
                    self.pretty_print_flops(output["flops_ts"])))
            else:
                print(" [*] Total FLOPs: {}.".format(
                    self.pretty_print_flops(output["flops_all"])))
            
        if write_gpu_time:
            accum_exec_time += exec_time
            print(" [*] GPU time {}".format(accum_exec_time))
            self.accum_exec_times.append(accum_exec_time)
            fmt_date = datetime.date.today().strftime("%d-%m-%Y")
            benchmark_file = os.path.join(self.save_root, "speed_benchmark_{}.txt".format(fmt_date))
            with open(benchmark_file, "a") as f:
                f.write("{:.6f}".format(accum_exec_time))

        return meta_img

    def get_exec_time_stats(self):
        mean = np.mean(self.accum_exec_times)
        std = np.std(self.accum_exec_times)
        return mean, std

    def save_results(self, meta_img):
        print(" [*] Saving results...")
        if meta_img is None:
            return
        self.save_meta_imgs(meta_img)
        self.cur_global_id += self.config.train_params.batch_size

    def save_meta_imgs(self, meta_img):

        # Center crop
        pad_h = (meta_img.shape[2] - self.target_height) // 2
        pad_w = (meta_img.shape[3] - self.target_width) // 2
        meta_img = meta_img[:, :, pad_h:pad_h+self.target_height, pad_w:pad_w+self.target_width]

        # Save the full image and the low-resolution image (for visualization)
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i

            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+".png")
            plt.imsave(save_path, meta_img_np[i])
    
            if hasattr(self.config.task, "lowres_height") and self.config.task.lowres_height > 0:
                lr_save_path = os.path.join(self.save_root, str(global_id).zfill(6)+"_lr.png")
                resize_ratio = self.config.task.lowres_height / self.config.task.height
                resize_shape  = (
                    int(round(self.target_height*resize_ratio)), 
                    int(round(self.target_width*resize_ratio)))
                lr_img = sk_resize((meta_img_np[i]*255).astype(np.uint8), resize_shape)
                plt.imsave(lr_save_path, lr_img)
