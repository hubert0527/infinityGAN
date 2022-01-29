import os
import yaml
import math
import shutil
import socket
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from easydict import EasyDict

import torch
from torch import nn

from test_managers.interactive_sampler import InteractiveSampler
from utils import *


def inference(g_ema, device, config, args):

    assert (not hasattr(config.train_params, "imp_feat_unfold")) or \
        (hasattr(config.train_params, "imp_feat_unfold") and config.train_params.imp_feat_unfold == False), \
        "Not carefully investigated!"
        
    """
    Setup env
    """
    exp_root = os.path.join(config.var.log_dir, config.var.exp_name)
    if args.debug:
        save_root = os.path.join(
            exp_root, 
            "test", 
            "debug-{}".format(config.task.config_name))
    else:
        save_root = os.path.join(
            exp_root, 
            "test", 
            "{}".format(config.task.config_name))
    if not os.path.exists(save_root): os.makedirs(save_root)
    shutil.copy2(
        config.task.config_path, 
        os.path.join(save_root, os.path.basename(config.task.config_path)))

    """
    Start inference
    """
    if config.task.num_gen == -1: # all, read number of images from dataset
        assert hasattr(config.task, "dataset_size"), \
            "Generate all should only be used for tasks with a real image dataset."
        config.task.num_gen = config.task.dataset_size
    iter_ = math.ceil(config.task.num_gen / config.task.batch_size)
    pbar = tqdm(
        range(iter_), 
        initial=0, 
        total=iter_, 
        dynamic_ncols=True, 
        smoothing=0.01)

    """
    Calculate shape and coordinates for patches
    """
    task_manager = import_func(config.task.task_manager)(g_ema, device, save_root, config)
    task_manager.task_specific_init()

    if args.inter_ckpt:
        if os.path.isfile(args.inter_ckpt):
            print(" [!] A single inter ckpt is loaded for all samples!")
            testing_vars_load_from = args.inter_ckpt
        else:
            testing_vars_load_from = sorted(glob(os.path.join(args.inter_ckpt, "*.pkl")))
    
    print(" [*] Setup complete, start testing!")
    for iter_ in pbar:

        if config.task.interactive or args.interactive:
            testing_vars = task_manager.create_vars(
                inv_records=args.inv_records, inv_placements=args.inv_placements)
            if args.inter_ckpt:
                if isinstance(testing_vars_load_from, list):
                    assert config.task.batch_size == 1, "Does not consider batch_size > 1 case!"
                    if task_manager.cur_global_id < len(testing_vars_load_from):
                        testing_vars.load(testing_vars_load_from[task_manager.cur_global_id])
                    else:
                        print(" [!] Run out of previous ckpt! Start from random!")
                else:
                    testing_vars.load(testing_vars_load_from)
            InteractiveSampler(task_manager, testing_vars, config)
        else:
            if args.speed_benchmark:
                assert (not hasattr(args, "inv_record")) or args.inv_record is None, "No fancy stuffs in benchmark."
                if iter_ < 10: # Do not use the stats from first ten, still unstable
                    task_manager.run_next(
                        save=False, write_gpu_time=False)
                else:
                    task_manager.run_next(
                        save=False, write_gpu_time=True)
            elif args.calc_flops:
                task_manager.run_next(save=False, write_gpu_time=False, calc_flops=True)
                if iter_ == 1: exit()
            else:
                task_manager.run_next(
                    save=True, write_gpu_time=False,
                    inv_records=args.inv_records, inv_placements=args.inv_placements)
        if args.debug:
            task_manager.exit()
            exit()

    task_manager.exit()

    if args.speed_benchmark:
        if not hasattr(config.task, "parallel_batch_size"):
            config.task.parallel_batch_size = -1
        exec_mean, exec_std = task_manager.get_exec_time_stats()
        print(" [*] Benchmark results over {} samples: {:.4f} +- {:.4f} (sec/image)".format(
            (config.task.num_gen - 1), exec_mean, exec_std))
        benchmark_record_path = "./logs-quant/benchmark_results/benchmark-{}.txt".format(socket.gethostname())
        if not os.path.exists(os.path.dirname(benchmark_record_path)):
            os.makedirs(os.path.dirname(benchmark_record_path))
        with open(benchmark_record_path, "a") as f:
            f.write("[-] EXP: Res {}x{} ; Parabatch {} ; {} GPUs\n".format(
                config.task.height, config.task.width, config.task.parallel_batch_size, config.var.n_gpu))
            f.write("{:.6f} +- {:.6f}\n".format(exec_mean, exec_std))
            f.write("\n")


if __name__ == "__main__":

    try:

        parser = argparse.ArgumentParser()
        parser.add_argument("--model-config", type=str)
        parser.add_argument("--test-config", type=str)

        parser.add_argument("--exp-suffix", type=str, default=None)
        parser.add_argument("--ckpt", type=str, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--interactive", action="store_true")
        parser.add_argument("--override-save-idx", type=int, default=None)

        parser.add_argument("--speed-benchmark", action="store_true")
        parser.add_argument("--calc-flops", action="store_true")

        # Flag for inversion distributed testing
        parser.add_argument("--inv-start-idx", type=int, default=None)
        parser.add_argument("--try-restrict-memory", type=float, default=1.)

        # Load from inversion
        parser.add_argument("--inv-records", type=str, default=None)
        parser.add_argument("--inv-placements", type=str, default=None)

        # Interactive recover from ckpt
        parser.add_argument("--inter-ckpt", type=str, default=None)

        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--archive-mode", action="store_true")
        parser.add_argument("--clear-fid-cache", action="store_true")
        args = parser.parse_args()

        if hasattr(torch.cuda.memory, "set_per_process_memory_fraction"):
            torch.cuda.memory.set_per_process_memory_fraction(args.try_restrict_memory)
            print(" [*] Set memory limit to {}%!".format(args.try_restrict_memory*100))

        """
        Parse Inv args
        """
        def parse_tuple(v):
            return tuple([float(vv) for vv in v.split(",")])
        if args.inv_records is not None:
            args.inv_records = [
                param for param in args.inv_records.split(":")]
            if args.inv_placements is None:
                args.inv_placements = (0.5, 0.5)
            else:
                args.inv_placements = [
                    parse_tuple(param) for param in args.inv_placements.split(":")]

        """
        Normal init
        """
        if args.verbose:
            def annoy_print(x):
                torch.cuda.synchronize()
                print(x, end="")
        else:
            annoy_print = dummy_func
        
        with open(args.model_config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config = EasyDict(config)
            config.var = EasyDict()
        config.var.exp_name = os.path.basename(args.model_config).split(".yaml")[0]
        print(" [*] Config {} loaded!".format(args.model_config))

        with open(args.test_config, "r") as f:
            config.task = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        config.task.config_name = os.path.basename(args.test_config).split(".yaml")[0]
        config.task.config_path = args.test_config
        config.task.exp_suffix = args.exp_suffix

        if args.inv_start_idx is not None:
            config.task.init_index = args.inv_start_idx
            assert config.task.batch_size == 1
            assert config.task.num_gen == 1
        elif args.override_save_idx is not None:
            config.task.init_index = args.override_save_idx
            assert config.task.batch_size == 1
            assert config.task.num_gen == 1
        else:
            config.task.init_index = 0

        if hasattr(config.task, "override_dataset_name"):
            print(" [!] Override dataset name to {} with specification in test-config!".format(
                config.task.override_dataset_name))
            config.data_params.dataset = config.task.override_dataset_name
        if hasattr(config.task, "override_dataset_data_size"):
            print(" [!] Override dataset raw resolution to {} with specification in test-config!".format(
                config.task.override_dataset_data_size))
            config.train_params.data_size = config.task.override_dataset_data_size
        if hasattr(config.task, "override_dataset_full_size"):
            print(" [!] Override dataset full-image resolution to {} with specification in test-config!".format(
                config.task.override_dataset_full_size))
            config.train_params.full_size = config.task.override_dataset_full_size

        if args.seed is not None:
            print(" [!] Forcingly use seed from cmdline!")
            cur_seed = args.seed
        elif config.task.seed is not None:
            print(" [!] Use default seed in task-config!!")
            cur_seed = config.task.seed
        else:
            print(" [!] Seed not specified, randomly assign one now!")
            cur_seed = np.random.randint(0, 9487)
        print(" [!] Current seed: {}".format(cur_seed))
        manually_seed(cur_seed)

        """
        Batch size calibration
        """
        if config.task.num_gen % config.task.batch_size != 0:
            bs = config.task.batch_size
            config.task.num_gen = math.ceil(config.task.num_gen / bs) * bs
            print(" [!] Force number of generated images to a multiple of batch size => {}".format(config.task.num_gen))
        config.train_params.batch_size = config.task.batch_size

        """
        Trait generation has specific num_gen
        """
        if hasattr(config.task, "is_trait_figure") and config.task.is_trait_figure:
            if hasattr(config.task, "n_trait_x"):
                config.task.num_gen = config.task.n_trait_x * config.task.n_trait_y
            else:
                config.task.num_gen = 1

        """
        Archive mode
        """
        if args.archive_mode:
            config.var.log_dir = "../../" # We are running in ./logs/<exp_name>/codes/
        else:
            config.var.log_dir = "./logs/"

        """
        Error file writing handling
            Remove previous error file (will make confusion on log synchronizing)
        """
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
                torch.backends.cudnn.benchmark = False
        else:
            raise ValueError(" [!] Please specify CUDA_VISIBLE_DEVICES!")

        # [NOTE] In debug mode:
        # 1. Will not write any logs
        # 2. Exit after first full iteration
        # 3. Force eval FID with one batch of fake samples; will not write FID cache if real stats are not exist
        if args.debug:
            print(" [Warning] Debug mode; Do not use this unless you know what you are doing!")
            bs = 1
            config.task.batch_size = bs * n_gpu
            config.log_params.n_save_sample = bs * n_gpu
        
        """
        Build G
        """
        g_ema = import_func(config.train_params.g_arch)(config=config)

        """
        Multi-GPU
        """
        if config.var.dataparallel:
            device = "cpu" # torch will auto do the GPU partitioning in backend
            g_ema = nn.DataParallel(g_ema).cuda()
        else:
            device = "cuda"
            g_ema = g_ema.to(device)
        g_ema.eval()


        """
        Load checkpoint
        """
        if args.ckpt is None:
            ckpt_dir = os.path.join(config.var.log_dir, config.var.exp_name, "ckpt")
            best_ckpt = os.path.join(ckpt_dir, "best_fid.pth.tar")

            assert os.path.exists(best_ckpt), "Cannot find checkpoint at {}!".format(best_ckpt)
            print(" [*] Found ckpt, load model from:", best_ckpt)
            ckpt = torch.load(best_ckpt, map_location=lambda storage, loc: storage)
        else:
            ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        safe_load_state_dict(g_ema, ckpt["g_ema"]) #, strict=False)
        print(" [*] Loaded ckpt at {} iter with FID {:.4f}".format(ckpt["iter"], ckpt["best_fid"]))

        with torch.no_grad():
            inference(g_ema, device, config, args)

    except Exception as e:
        if e is not KeyboardInterrupt:
            error_f = os.path.join(config.var.log_dir, config.var.exp_name, "test-error-log.txt")
            with FileLock(error_f, timeout=10, delay=0.1) as lock:
                with open(error_f, "w+") as f:
                    f.write(str(e) + "\n")
                    f.write(" *** stack trace *** \n")
                    f.write(traceback.format_exc())
        raise e
