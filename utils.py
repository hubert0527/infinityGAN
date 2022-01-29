import os
import re
import gc
import time
import yaml
import math
import random
import shutil
import argparse
import importlib
import traceback
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw 
from tqdm import tqdm
from glob import glob
# from easydict import EasyDict
from tensorboardX import SummaryWriter

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils

from dataset import DictTensor


def find_font_source():
    default_path = "./assets/OpenSans-Bold.ttf"
    dev_path = "~/infinityGAN/infinityGAN/asserts/OpenSans-Bold.ttf"

    if os.path.exists(default_path):
        return default_path
    elif os.path.exists(dev_path):
        return dev_path
    else:
        return None


def purge_cache():
    n = gc.collect()
    torch.cuda.empty_cache()


def vis_structure_latent_slice(structure_latent_slice, slot_size=64, font_size=22, normalize_value=True):
    # spatial_latent_slice shape: (H, W)
    font_file_path = find_font_source()

    min_, max_ = structure_latent_slice.min(), structure_latent_slice.max()
    structure_latent_slice_norm = (structure_latent_slice - min_) / (max_ - min_) # normalize to (0, 1)
    structure_latent_slice_norm = structure_latent_slice_norm * 0.8 + 0.1

    if normalize_value:
        text_src = structure_latent_slice_norm
        color_src = structure_latent_slice_norm
    else:
        text_src = structure_latent_slice
        color_src = structure_latent_slice_norm

    imgs = []
    for x in range(structure_latent_slice.shape[0]):
        for y in range(structure_latent_slice.shape[1]):
            face_color = round(color_src[x,y].item(), 2)
            text_score = round(text_src[x,y].item(), 2)
            text_c = 255 if face_color<0.5 else 0

            img = Image.fromarray((np.ones((slot_size, slot_size, 3))*255*face_color).astype(np.uint8))
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_file_path, font_size)
            draw.text((slot_size//8, slot_size//4), "{: .2f}".format(text_score), (text_c, text_c, text_c), font=font)
            imgs.append(np.array(img))
    imgs = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)).float()

    meta_img = utils.make_grid(
        imgs,
        nrow=structure_latent_slice.shape[1],
        normalize=True,
        range=(0, 255),
    )
    return meta_img


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    if isinstance(model1, nn.DataParallel):
        device = next(model1.module.parameters()).device
    else:
        device = next(model1.parameters()).device
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data.to(device), alpha=1 - decay)


class SequentialSamplerWithInit(data.SequentialSampler):
    def __init__(self, data_source, init_index=0):
        self.data_source = data_source
        self.init_index = init_index

    def __iter__(self):
        indices = list(range(self.init_index, len(self.data_source))) + \
                  list(range(self.init_index)) 
        # Round back to 0, avoid potential bugs that someone forgets to turn it off at the final evaluation
        return iter(indices)


def data_sampler(dataset, shuffle, init_index=0):
    if shuffle:
        assert init_index==0, "Shuffle mode does not support init_index!"
        return data.RandomSampler(dataset)
    else:
        return SequentialSamplerWithInit(dataset, init_index=init_index)


def requires_grad(model, flag=True):
    if model is None: return
    for p in model.parameters():
        p.requires_grad = flag


def make_nonstopping(loader):
    while True:
        for batch in loader:
            yield batch


def rm_outdated_ckpt(pattern=None, max_to_keep=2):
    assert pattern is not None
    paths = sorted(glob(pattern))
    # iter_s = [re.search("[0-9]+", path).group(0) for path in paths]
    for path in paths[:-max_to_keep]:
        os.remove(path)


def unify_local_latent_ext(local_latent, local_latent_ext_list):
    l = []
    largest = local_latent_ext_list[-1]
    lB, lC, lH, lW = largest.shape

    B, C, H, W = local_latent.shape
    pad_h = (lH - H) // 2
    pad_w = (lW - W) // 2
    local_latent = largest[:, :, pad_h:pad_h+H, pad_w:pad_w+W]

    for v in local_latent_ext_list:
        B, C, H, W = v.shape
        pad_h = (lH - H) // 2
        pad_w = (lW - W) // 2
        l.append(largest[:, :, pad_h:pad_h+H, pad_w:pad_w+W])
    return local_latent, l


def dummy_func(*args, **kwargs):
    return None


class IdleWriter():
    def __init__(self):
        self.idle_func = dummy_func
    def __getattr__(self, attr):
        return self.idle_func


def auto_batched_inference(g_ema, config, partition_keys, *args, **kwargs):
    assert len(args) == 0
    batch_size = config.train_params.batch_size
    n_samples = config.log_params.n_save_sample
    n_batches = config.log_params.n_save_sample // batch_size
    if config.log_params.n_save_sample % batch_size != 0:
        n_batches += 1

    for k in partition_keys:
        assert k in kwargs, \
            "Expected all partition keys are specified in kwargs ({}), but cannot find {}!".format(kwargs.keys(), k)

    def partition(k, v, st, ed):
        if not (k in partition_keys):
            return v
        elif isinstance(v, DictTensor):
            dict_template = {}
            for kk,vv in v.items():
                if isinstance(vv, torch.Tensor) and len(vv.shape) > 0:
                    dict_template[kk] = vv[st:ed]
                else:
                    dict_template[kk] = vv
            return DictTensor(**dict_template)
        elif v is None:
            return v
        else:
            return v[st:ed]

    all_ret = []
    for i in range(n_batches):
        st, ed = i*batch_size, (i+1)*batch_size
        batch_kwargs = {
            k: partition(k,v,st,ed) for k,v in kwargs.items()}
        all_ret.append(
            g_ema(**batch_kwargs).cpu().detach())
    all_ret = all_ret[:n_samples]

    if isinstance(all_ret[0], torch.Tensor):
        return torch.cat(all_ret, 0)
    else:
        ret_tensor_dict = {}
        for k,v in all_ret[0].items():
            if isinstance(v, torch.Tensor):
                if len(v.shape) == 0: # zero-dimensional tensor
                    ret_tensor_dict[k] = torch.mean(torch.stack([el[k] for el in all_ret]))
                else:
                    ret_tensor_dict[k] = torch.cat([el[k] for el in all_ret], 0)
            elif isinstance(v, np.ndarray):
                ret_tensor_dict[k] = np.concatenate([el[k] for el in all_ret], 0)
            elif isinstance(v, list):
                ret_tensor_dict[k] = sum([el[k] for el in all_ret]) 
            elif v is None:
                ret_tensor_dict[k] = None
            else:
                raise NotImplementedError("Return value with type {} is not handled!".format(type(v)))
        return DictTensor(ret_tensor_dict)


def import_func(path):
    toks = path.split(".")
    module = ".".join(toks[:-1])
    func = toks[-1]
    return getattr(importlib.import_module(module), func)


def safe_load_state_dict(model, state_dict, strict=True):
    # FrEAI saves some unused runtime variables
    state_dict = {k:v for k,v in state_dict.items() if 'tmp_var' not in k}
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict, strict=strict)
    elif isinstance(model, nn.Module):
        model.load_state_dict(state_dict, strict=strict)
    else: # Optimizer
        model.load_state_dict(state_dict)


def manually_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

# Codes from:
# https://github.com/dmfrey/FileLock/blob/master/filelock/filelock.py

import os
import time
import errno
 
class FileLockException(Exception):
    pass
 
class FileLock(object):
    """ A file locking mechanism that has context-manager support so 
        you can use it in a with statement. This should be relatively cross
        compatible as it doesn't rely on msvcrt or fcntl for the locking.
    """
 
    def __init__(self, file_name, timeout=10, delay=.05):
        """ Prepare the file locker. Specify the file to lock and optionally
            the maximum timeout and the delay between each attempt to lock.
        """
        if timeout is not None and delay is None:
            raise ValueError("If timeout is not None, then delay must not be None.")
        self.is_locked = False
        self.lockfile = os.path.join(os.getcwd(), "%s.lock" % file_name)
        self.file_name = file_name
        self.timeout = timeout
        self.delay = delay
 
 
    def acquire(self):
        """ Acquire the lock, if possible. If the lock is in use, it check again
            every `wait` seconds. It does this until it either gets the lock or
            exceeds `timeout` number of seconds, in which case it throws 
            an exception.
        """
        start_time = time.time()
        while True:
            try:
                self.fd = os.open(self.lockfile, os.O_CREAT|os.O_EXCL|os.O_RDWR)
                self.is_locked = True #moved to ensure tag only when locked
                cur_time = time.time()
                time_diff = int(cur_time - start_time)
                if time_diff > 100 and time_diff % 60 == 0:
                    print(" [!] Cannot acquire lock {} after {} seconds".format(self.lockfile, time_diff))
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                if self.timeout is None:
                    raise FileLockException("Could not acquire lock on {}".format(self.file_name))
                if (time.time() - start_time) >= self.timeout:
                    raise FileLockException("Timeout occured.")
                time.sleep(self.delay)
#        self.is_locked = True
 
 
    def release(self):
        """ Get rid of the lock by deleting the lockfile. 
            When working in a `with` statement, this gets automatically 
            called at the end.
        """
        if self.is_locked:
            os.close(self.fd)
            os.unlink(self.lockfile)
            self.is_locked = False
 
 
    def __enter__(self):
        """ Activated when used in the with statement. 
            Should automatically acquire a lock to be used in the with block.
        """
        if not self.is_locked:
            self.acquire()
        return self
 
 
    def __exit__(self, type, value, traceback):
        """ Activated at the end of the with statement.
            It automatically releases the lock if it isn't locked.
        """
        if self.is_locked:
            self.release()
 
 
    def __del__(self):
        """ Make sure that the FileLock instance doesn't leave a lockfile
            lying around.
        """
        self.release()


def log_memory(config, writer, iter_):

    for gpu_id in range(config.var.n_gpu):
        writer.add_scalar("mem_gpu{}/cur-allocated".format(gpu_id), 
            torch.cuda.memory_allocated()/1024/1024, iter_)
        writer.add_scalar("mem_gpu{}/max-allocated".format(gpu_id), 
            torch.cuda.max_memory_allocated()/1024/1024, iter_)

        writer.add_scalar("mem_gpu{}/cur-reserved".format(gpu_id), 
            torch.cuda.memory_reserved()/1024/1024, iter_)
        writer.add_scalar("mem_gpu{}/max-reserved".format(gpu_id), 
            torch.cuda.max_memory_reserved()/1024/1024, iter_)

        writer.add_scalar("mem_gpu{}/cur-cached".format(gpu_id), 
            torch.cuda.memory_cached()/1024/1024, iter_)
        writer.add_scalar("mem_gpu{}/max-cached".format(gpu_id), 
            torch.cuda.max_memory_cached()/1024/1024, iter_)
