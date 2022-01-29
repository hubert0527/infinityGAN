import os
import time
import yaml
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from glob import glob
from easydict import EasyDict

import torch
from torch import nn

torch.backends.cudnn.benchmark = True
#torch.autograd.set_detect_anomaly(True)

from dataset import MultiResolutionDataset
from libs.fid import eval_fid
from libs.inception_score import inception_score
from libs.calc_inception import load_patched_inception_v3
from latent_sampler import LatentSampler
from utils import *
from quant_eval_utils import QuantEvalSampleGenerator, QuantEvalDataLoader

SET_TEST_ID = True # Making testing-time randomized noise inputs from StyleGAN2 fixed
TB_PARTITION_STEPS = 100000 # Partition event file for efficient rsync


def eval(args, latent_sampler, g_ema, inception, device, config):

    if g_ema is not None: 
        g_ema.eval()

    """
    Cast FID calculation spec
    """
    if hasattr(config.train_params, "extra_pre_resize"):
        real_data_res = config.train_params.extra_pre_resize
    else: # StyleGAN2 baseline
        assert config.train_params.styleGAN2_baseline
        real_data_res = config.train_params.full_size
    assert real_data_res in {128, 256}, "In this paper, we only benchmark in size {128, 256}. Got {}.".format(real_data_res)

    eval_gen_res = real_data_res * args.scale

    # InfinityGAN is trained with larger image, so the same resolution equivalents to smaller FoV.
    # Here, we ensures the FoV is the same as the StyleGAN2 baseline
    fov_scale = config.train_params.full_size / real_data_res
    raw_gen_res = int(np.ceil(eval_gen_res * fov_scale))

    if args.seq_inference:
        assert (not hasattr(config.train_params, "styleGAN2_baseline")) or (not config.train_params.styleGAN2_baseline)
        assert args.scale > 1, "Set sequential inference with scale==1 is meaningless"
        use_seq_inf = True
    else:
        use_seq_inf = False

    """
    Create dataloader and generator
    """
    if args.img_folder is not None:
        postprocessing_params = [
            ["assert", eval_gen_res],
            ["resize", real_data_res],
        ]
    else:
        postprocessing_params = [
            ["scale", 1 / fov_scale],
            ["crop", eval_gen_res],
            ["resize", real_data_res],
        ]
    fake_generator = \
        QuantEvalSampleGenerator(
            g_ema, 
            latent_sampler, 
            img_folder=args.img_folder, # if applicable
            output_size=raw_gen_res, 
            use_seq_inf=use_seq_inf,
            postprocessing_params=postprocessing_params,
            fid_type=args.type,
            device=device, 
            config=config,
            use_pil_resize=args.use_pil_resize)


    stats_key = "benchmark-{}-{}-RealRes{}".format(
        args.type, config.data_params.dataset, real_data_res)
    # FID statistics can be different for different PyTorch version, not sure about cuda
    stats_key += f"_PT{torch.__version__}_cu{torch.version.cuda}"
    fid_cache_path = os.path.join(".fid-cache/", stats_key+".pkl")
    if os.path.exists(fid_cache_path):
        if args.clear_fid_cache:
            os.remove(fid_cache_path)
            use_cache = False
        else:
            use_cache = True
    else:
        use_cache = False

    if not use_cache:
        dataset = MultiResolutionDataset(
            split="train",
            config=config,
            is_training=False,
            # return "full" of real full images and crop on-the-fly
            disable_extra_cropping=True,
            simple_return_full=True,
            override_full_size=real_data_res) 
        real_dataloader = QuantEvalDataLoader(dataset, real_data_res, device, config)
    else:
        real_dataloader = None

    """
    Eval
    """
    st = time.time()
    if args.metric == "is":
        assert args.scale == 1, "We didn't implement scaleinv IS."
        n_batch = int(np.ceil(config.test_params.n_fid_sample / config.train_params.batch_size))
        all_imgs = []
        for img_batch in tqdm(fake_generator(n_batch), total=n_batch):
            img_batch = ((img_batch + 1) / 2).cpu() # [-1, 1] => [0, 1]
            all_imgs.append(img_batch)
        all_imgs = torch.cat(all_imgs, 0)
        is_mean, is_std = inception_score(all_imgs, device="cuda", batch_size=config.train_params.batch_size, resize=False, splits=10)
        print(" [*] IS time spend {}".format(args.type, time.time()-st))
        print(" [*] IS at eval_gen_res {} is {}+-{} (ckpt patch FID = {})".format(
            eval_gen_res, is_mean, is_std, config.var.best_fid))
    elif args.metric == "fid":
        if args.type == "spatial":
            fid = eval_fid(
                real_dataloader, fake_generator, inception, stats_key, None, device, config, 
                spatial_partition_cat=True, assert_eval_shape=real_data_res)
        elif args.type in {"scaleinv", "alis"}:
            fid = eval_fid(
                real_dataloader, fake_generator, inception, stats_key, None, device, config, 
                spatial_partition_cat=False, assert_eval_shape=real_data_res)
        else:
            raise NotImplementedError("Unknown FID variant {}".format(args.type))
        print(" [*] {} FID time spend {}".format(args.type, time.time()-st))
        print(" [*] FID (type {}) at eval_gen_res {} is {} (ckpt patch FID = {})".format(
            args.type, eval_gen_res, fid, config.var.best_fid))

    """
    Setup Logging
    """
    if args.metric == "is":
        log_root = os.path.join("logs-quant", "IS")
        filename = f"EvalGenRes{eval_gen_res}-Exp-{config.var.exp_name}.txt"
        score = "{:.6f}+-{:.6f}\n".format(is_mean, is_std)
    else:
        log_root = os.path.join("logs-quant", "FID-"+args.type)
        filename = f"Scale{args.scale}-EvalGenRes{eval_gen_res}-Exp-{config.var.exp_name}.txt"
        score = "{:.6f}\n".format(fid)

    if not os.path.exists(log_root):
        os.makedirs(log_root)
    with open(os.path.join(log_root, filename), "a") as lf:
        lf.write(score)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--type", type=str, choices=["default", "scaleinv", "spatial", "alis"], default="default")
    parser.add_argument("--seq-inference", action="store_true")
    parser.add_argument("--alis-config", type=str, default=None)

    parser.add_argument("--metric", type=str, choices=["fid", "is"], default="fid")

    # Other evaluation methods other than <exp_name>/ckpt/best_fid.pth.tar
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--img-folder", type=str, default=None)
    parser.add_argument("--use-pil-resize", action="store_true")

    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--archive-mode", action="store_true", default=False)
    parser.add_argument("--clear-fid-cache", action="store_true", default=False)
    args = parser.parse_args()

    if args.verbose:
        def annoy_print(x):
            torch.cuda.synchronize()
            print(x, end="")
    else:
        annoy_print = dummy_func
    
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(config)
        config.var = EasyDict()
    config.var.exp_name = os.path.basename(args.config).split(".yaml")[0]
    print(" [*] Config {} loaded!".format(args.config))

    if args.type == "alis":
        assert args.alis_config is not None, "Evaluate alis fid must specify a config!"
        with open(args.alis_config, "r") as f:
            config.task = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        config.task.config_name = os.path.basename(args.alis_config).split(".yaml")[0]
        config.task.config_path = args.alis_config
        config.train_params.batch_size = config.task.batch_size # Usually 1, never tested other cases.
        config.task.save_type = "patches-centercrop" # FID requires shape-aligned patches
        print(" [*] ALIS eval config {} loaded!".format(args.alis_config))

    if args.archive_mode:
        config.var.log_dir = "../../" # We are running in ./logs/<exp_name>/codes/
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
    else:
        raise ValueError(" [!] Please specify CUDA_VISIBLE_DEVICES!")

    if args.batch_size is not None:
        config.train_params.batch_size = args.batch_size
    
    try:
        if args.img_folder:
            g_ema = None
            latent_sampler = None
            config.var.best_fid = -1
        else:
            """
            Build G & D
            """
            g_ema = import_func(config.train_params.g_arch)(config=config)
            latent_sampler = LatentSampler(g_ema, config)

            """
            Multi-GPU
            """
            if config.var.dataparallel:
                device = "cpu" # torch will auto do the GPU partitioning in backend
                g_ema = nn.DataParallel(g_ema).cuda()
            else:
                device = "cuda"
                g_ema = g_ema.to(device)

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
            config.var.best_fid = ckpt["best_fid"]

        
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
            inception = nn.DataParallel(inception).cuda()
        else:
            device = "cuda"
            inception = inception.to(device)

        eval(args, latent_sampler, g_ema, inception, device, config)
    except Exception as e:
        if e is not KeyboardInterrupt:
            error_dirs = sorted(glob("./burst-errors-*"))
            error_f = os.path.join(config.var.log_dir, config.var.exp_name, "error-log.txt")
            with open(error_f, "w") as f:
                f.write(str(e) + "\n")
                f.write(" *** stack trace *** \n")
                f.write(traceback.format_exc())
        raise e
