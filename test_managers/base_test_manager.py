import math
import time
import numpy as np

import torch

from coord_handler import CoordHandler
from latent_sampler import LatentSampler


class BaseTestManager():

    def __init__(self, g_ema, device, save_root, config):
        self.g_ema = g_ema.eval()
        self.device = device
        self.save_root = save_root
        self.config = config

        self.accum_exec_times = []

        if config.var.dataparallel:
            self.g_ema_module = g_ema.module
        else:
            self.g_ema_module = g_ema

        self.ss_unfold_size = self.config.train_params.ss_n_layers * self.config.train_params.ss_unfold_radius

        if hasattr(config.task, "parallel_batch_size") and config.task.parallel_batch_size != config.task.batch_size:
            print(" [!] Parallel batched generation is enabled!")
            assert config.task.batch_size == 1, "Though should supported, but never tested."
            self.enable_parallel_batching = True
            self.parallel_stack = []
        else:
            self.enable_parallel_batching = False

        self.coord_handler = CoordHandler(config)
        self.latent_sampler = LatentSampler(g_ema, config)

        self.coord_handler.eval()

        if hasattr(self.config.task, "init_index") and self.config.task.init_index is not None:
            self.cur_global_id = self.config.task.init_index
        else:
            self.cur_global_id = 0
        ts_input_size = config.train_params.ts_input_size

        # One can easily confused here (even I frequently confuse myself...). The 
        #  two images generated with two consecutive blocks of spatial_latent is NOT
        #  direct neighbors of each other. There is a missing region between them that
        #  was omitted due to the no-padding "odd-architecture" of the generator. More
        #  specifically, some of the intermediate features are thrown away, as it is only
        #  used to avoid the generator padding.
        # 
        # Here we manually compute how many pixels does it displaced
        #  while moving one pixel in the latent space. Then calculate
        #  how many pixels shall we displace in practice in order to 
        #  generate spatially consecutive patches.
        #
        # [Example]
        # Original arch: 5  =>  7 => 11 => 19 => 17 (assume end here)
        # Doubled arch:  10 => 17 => 31 => 59 => 57
        # pixel_space_disp = 57 - 17 = 40
        #
        #     Z: |   5    | (omitted_area) |   5    |
        #     X: |   17   |       23       |   17   |
        #    G1: |   17   |
        #    G2:   (16)  |   17   |
        #    G3:   (16)    (16)  |   17   |
        #
        # pixelspace_disp_unit  = 40 // 5 = 8 
        #     (N pixels disp. in pixel space per 1 pixel disp. in latent space)
        # latentspace_step_size = 17 // 8 = 2
        # pixelspace_step_size  = 2 * 8 = 16
        #
        # [Notes] 
        # 1. The overlapping area between G1, G2 and G3 shall be consistent with our calculation
        # 2. The GNN part "MAY" not truely-fully-convolutional, need to specifically dealt with later.
        # 3. The GNN padding is is a constant, and automatically discounted while computing `outfeat_disps`.
        self.outfeat_sizes_list = \
            self._get_feature_sizes(side="output", multiplier=1, include_ss=False)
        self.outfeat_sizes_list_doubled  = \
            self._get_feature_sizes(side="output", multiplier=2, include_ss=False)
        outfeat_disps = self.outfeat_sizes_list_doubled - self.outfeat_sizes_list
        assert (outfeat_disps % ts_input_size == 0).all(), \
            "By definition, ts_input_size {} should be a divisor of {}".format(
                ts_input_size, outfeat_disps.tolist())

        # Similarly, process input space 
        # (actually, it is just `np.roll(outfeat_disps, -1)[:-1]`, but I am too lazy to impl. `include_ss` in TS...)
        self.infeat_sizes_list = self._get_feature_sizes(
            side="input", 
            output_size=self.outfeat_sizes_list[-1], 
            include_ss=False)
        self.in_size_doubled_list = self._get_feature_sizes(
            side="input", 
            output_size=self.outfeat_sizes_list_doubled[-1],
            include_ss=False)
        infeat_disps = self.in_size_doubled_list - self.infeat_sizes_list
        assert (infeat_disps % ts_input_size == 0).all(), \
            "By definition, ts_input_size {} should be a divisor of {}".format(
                ts_input_size, infeat_disps.tolist())
        
        # Get the image space / latent space step size
        pixelspace_disp_unit = outfeat_disps[-1] // ts_input_size
        self.pixelspace_step_size  = (self.outfeat_sizes_list[-1] // pixelspace_disp_unit) * pixelspace_disp_unit
        self.latentspace_step_size = self.pixelspace_step_size // pixelspace_disp_unit

        # Consequently, also records the step sizes in the feature space (input-side) for further noise/latent handling
        infeat_disp_units = infeat_disps // ts_input_size
        self.infeat_step_sizes = self.latentspace_step_size * infeat_disp_units

        # Consequently, also records the step sizes in the feature space (output-side) for further noise/latent handling
        outfeat_disp_units = outfeat_disps // ts_input_size
        self.outfeat_step_sizes = self.latentspace_step_size * outfeat_disp_units

        # Double-check the correctness of the spec in the config
        if config.train_params.training_modality == "patch":
            out_res_spec = config.train_params.patch_size
        else:
            raise ValueError("Unknown training modality {}".format(config.train_params.training_modality))
        assert self.outfeat_sizes_list[-1] == out_res_spec, \
            "In general, in terms of efficiency, I should've set generator output" +\
            " resolution config ({}) equals to the true generator output resolution ({}).".format(
                out_res_spec, self.outfeat_sizes_list[-1])

    def _get_feature_sizes(self, side=None, multiplier=1, output_size=None, include_ss=False):
        assert side in {"input", "output"}
        if side == "input":
            assert output_size is not None
            assert multiplier == 1
            size_list = self.g_ema_module.calc_in_spatial_size(
                output_size, include_ss=include_ss, return_list=True)
        elif side == "output":
            assert include_ss == False, "SS is never used, some logics are not implemented/verified."
            size_list = self.g_ema_module.calc_out_spatial_size(
                self.config.train_params.ts_input_size*multiplier, include_ss=include_ss, return_list=True)
        size_list = np.array(size_list).astype(np.uint32)
        return size_list

    def task_specific_init(self):
        # Task-specific initialization
        raise NotImplementedError()

    def run_next(self):
        # Run the complete task once
        raise NotImplementedError()

    def save_results(self, path):
        raise NotImplementedError()

    def exit(self):
        return # Usually does nothing

    def get_exec_time_stats(self):
        mean = np.mean(self.accum_exec_times)
        std = np.std(self.accum_exec_times)
        return mean, std

    def pretty_print_flops(self, flops):
        ret = ""
        if flops > 1e12:
            ret += str(int((flops//1e12) % 1000)).zfill(3) + "T "
        if flops > 1e9:
            ret += str(int((flops//1e9) % 1000)).zfill(3) + "G "
        if flops > 1e6:
            ret += str(int((flops//1e6) % 1000)).zfill(3) + "M "
        if flops > 1e3:
            ret += str(int((flops//1e3) % 1000)).zfill(3) + "K "
        if flops > 1:
            ret += str(int((flops//1e1) % 1000)).zfill(3)
        return ret

    """
    Parallel batching
    """
    def agg_args(self, args):
        if isinstance(args[0], dict):
            raise NotImplementedError("Unused")
            # return {k: self.parallel_agg_kwargs(v) for k,v in kwargs.items()}
        elif isinstance(args[0], list):
            # return [self.parallel_agg_kwargs(v) for v in kwargs]
            ret = []
            for i in range(len(args[0])):
                ret.append(self.agg_args([v[i] for v in args]))
            return ret
        elif isinstance(args[0], torch.Tensor):
            return torch.cat(args, 0)
        elif isinstance(args[0], bool):
            return args[0]
        else: # Other scalar types, assume all shares the same value
            assert all([v==args[0] for v in args]), \
                "Assume all params within the parallel batch share the same scalar value, but got {}".format(args)
            return args[0]

    def parallel_agg_kwargs(self, kwargs_list):
        all_keys = [k for k in kwargs_list[0].keys()]
        ret = {}
        for k in all_keys:
            ret[k] = self.agg_args([kwargs[k] for kwargs in kwargs_list])
        return ret

    def ensure_contiguous(self, kwargs):
        if isinstance(kwargs, dict):
            return {k: self.ensure_contiguous(v) for k,v in kwargs.items()}
        elif isinstance(kwargs, list):
            return [self.ensure_contiguous(v) for v in kwargs]
        elif isinstance(kwargs, torch.Tensor):
            return kwargs.contiguous()
        else:
            return kwargs

    def maybe_parallel_inference(self, testing_vars, g_ema_kwargs=None, index_tuple=None, 
                                 flush=False, return_exec_time=False, calc_flops=False):
        exec_time = -1
        flops = {"all": 0, "ss": 0, "ts": 0}
        if self.enable_parallel_batching:
            if g_ema_kwargs is not None:
                self.parallel_stack.append({
                    "g_ema_kwargs": g_ema_kwargs,
                    "index_tuple": index_tuple
                })
            else:
                assert flush, "Only expected use case."

            if flush and len(self.parallel_stack) == 0:
                return exec_time, flops
            elif flush or len(self.parallel_stack) == self.config.task.parallel_batch_size:
                all_g_ema_kwargs = [d["g_ema_kwargs"] for d in self.parallel_stack]
                agg_g_ema_kwargs = self.parallel_agg_kwargs(all_g_ema_kwargs)
                agg_g_ema_kwargs = self.ensure_contiguous(agg_g_ema_kwargs)
                if return_exec_time:
                    torch.cuda.synchronize()
                    start_time = time.time()
                output = self.g_ema(**agg_g_ema_kwargs)
                if return_exec_time:
                    torch.cuda.synchronize()
                    exec_time = time.time() - start_time
                if calc_flops:
                    assert self.config.train_params.batch_size == 1, \
                        "Does not consider batch_size!=1 and parallel batching happens at the same time!"
                    pbatch_size = output["gen"].shape[0]
                    flops = {
                        "all": output["flops_all"].sum() * pbatch_size,
                        "ss": output["flops_ss"].sum() * pbatch_size,
                        "ts": output["flops_ts"].sum() * pbatch_size,
                    }
                patch = output["gen"]
                for i,d in enumerate(self.parallel_stack):
                    img_x_st, img_x_ed, img_y_st, img_y_ed = d["index_tuple"]
                    testing_vars.meta_img[:, :, img_x_st:img_x_ed, img_y_st:img_y_ed] = patch[i:i+1].detach().cpu()
                self.parallel_stack = []
            else:
                exec_time = 0
                pass # Wait until queue is full or forced flush
        else:
            if flush: 
                return exec_time, flops # Normal mode does not have flushing issue
            img_x_st, img_x_ed, img_y_st, img_y_ed = index_tuple
            g_ema_kwargs = self.ensure_contiguous(g_ema_kwargs)
            if return_exec_time:
                torch.cuda.synchronize()
                start_time = time.time()
            output = self.g_ema(**g_ema_kwargs)
            if return_exec_time:
                torch.cuda.synchronize()
                exec_time = time.time() - start_time
            if calc_flops:
                flops = {
                    "all": output["flops_all"],
                    "ss": output["flops_ss"],
                    "ts": output["flops_ts"],
                }
            patch = output["gen"]
            testing_vars.meta_img[:, :, img_x_st:img_x_ed, img_y_st:img_y_ed] = patch.detach().cpu()
        return exec_time, flops

    """
    Interactive utils
    """
    def is_overlaping_update_map(self, update_map, xst, xed, yst, yed):
        return (update_map[:, :, xst:xed, yst:yed] > 0).any()

