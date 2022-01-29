import os
import lmdb
import yaml
import math
import socket
import argparse
import numpy as np
from io import BytesIO
from PIL import Image
from glob import glob
from tqdm import tqdm
from easydict import EasyDict
from random import randrange

import torch
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms

from env_config import LMDB_ROOTS

from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024**2)


def safe_randrange(low, high):
    if low==high:
        return low
    else:
        return randrange(low, high)


class DictTensor(dict):

    def to(self, device):
        new_self = DictTensor()
        for k,v in self.items():
            if isinstance(v, torch.Tensor):
                new_self[k] = v.to(device)
            else:
                new_self[k] = v
        return new_self

    def cpu(self):
        new_self = DictTensor()
        for k,v in self.items():
            if isinstance(v, torch.Tensor):
                new_self[k] = v.cpu()
            else:
                new_self[k] = v
        return new_self

    def detach(self):
        new_self = DictTensor()
        for k,v in self.items():
            if isinstance(v, torch.Tensor):
                new_self[k] = v.detach()
            else:
                new_self[k] = v
        return new_self

    def get_device(self):
        return list(self.values())[0].device

    def __setattr__(self, attr, value):
        if attr == "requires_grad":
            for v in self.values():
                # Note: Tensor with non-float type cannot requires grad
                if isinstance(v, torch.Tensor) and v.dtype not in {torch.int32, torch.int64}:
                    v.requires_grad = value
            #for v in self.attrs.values():
            #    v.requires_grad = value
        else:
            super().__setattr__(attr, value)


class MaybeCenterCrop():
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        if self.crop_size is None:
            return img
        else:
            pad_h = (img.height - self.crop_size) // 2
            pad_w = (img.width - self.crop_size) // 2
            img = img.crop((pad_w, pad_h, pad_w+self.crop_size, pad_h+self.crop_size))
            return img


class MaybeResize():
    def __init__(self, full_size):
        self.full_size = full_size

    def __call__(self, img):
        if img.width == self.full_size and img.height==self.full_size:
            return img
        else:
            if img.height != img.width:
                if img.height > img.width:
                    pad_h = (img.height - img.width) // 2
                    pad_w = 0
                    size = img.width
                else:
                    pad_h = 0
                    pad_w = (img.width - img.height) // 2
                    size = img.height
                img = img.crop((pad_w, pad_h, pad_w+size, pad_h+size))
            assert img.height == img.width
            return img.resize([self.full_size, self.full_size], resample=Image.LANCZOS)


class CropPatch():
    def __init__(self, input_size, patch_size, config,
                 random_crop=False, center_crop=False, disable_ac=False):
        assert (random_crop or center_crop) and not (random_crop and center_crop)
        self.input_size = input_size
        self.patch_size = patch_size
        self.config = config

        self.random_crop = random_crop
        self.center_crop = center_crop

        if disable_ac:
            self.return_ac_coords = False
        elif self.input_size != self.patch_size:
            self.return_ac_coords = True
        else:
            self.return_ac_coords = False # always 1, meaningless, will randomly sample outside

        self.skip_cropping = (self.input_size == self.patch_size)

    def ac_coords_from_ratio(self, coord_ratio, proj):
        coord_ratio = coord_ratio * 2 - 1 # [-1, 1]
        if proj == "raw":
            return coord_ratio
        # elif proj == "tanh":
        #     return math.tanh(coord_ratio)
        elif proj == "sin":
            return math.sin(coord_ratio * math.pi)
        elif proj == "cos":
            return math.cos(coord_ratio * math.pi)
        else:
            raise ValueError("Unknown proj {}".format(proj))

    def __call__(self, img):

        assert img.size == (self.input_size, self.input_size)
        if self.skip_cropping:
            return img, None

        if self.random_crop:
            xst = safe_randrange(0, self.input_size - self.patch_size)
            yst = safe_randrange(0, self.input_size - self.patch_size)
            if self.return_ac_coords:
                if self.config.train_params.coord_num_dir == 1:
                    ac_coords = torch.FloatTensor([
                        self.ac_coords_from_ratio(xst / (self.input_size - self.patch_size - 1), "raw"),
                    ])
                elif self.config.train_params.coord_num_dir == 2:
                    ac_coords = torch.FloatTensor([
                        self.ac_coords_from_ratio(yst / (self.input_size - self.patch_size - 1), "sin"),
                        self.ac_coords_from_ratio(yst / (self.input_size - self.patch_size - 1), "cos"),
                    ])
                elif self.config.train_params.coord_num_dir == 4:
                    ac_coords = torch.FloatTensor([
                        self.ac_coords_from_ratio(xst / (self.input_size - self.patch_size - 1), "sin"),
                        self.ac_coords_from_ratio(xst / (self.input_size - self.patch_size - 1), "cos"),
                        self.ac_coords_from_ratio(yst / (self.input_size - self.patch_size - 1), "sin"),
                        self.ac_coords_from_ratio(yst / (self.input_size - self.patch_size - 1), "cos"),
                    ])
                elif self.config.train_params.coord_num_dir in {3, 21}:
                    ac_coords = torch.FloatTensor([
                        self.ac_coords_from_ratio(xst / (self.input_size - self.patch_size - 1), "raw"),
                        self.ac_coords_from_ratio(yst / (self.input_size - self.patch_size - 1), "sin"),
                        self.ac_coords_from_ratio(yst / (self.input_size - self.patch_size - 1), "cos"),
                    ])
                else:
                    raise ValueError("Unknown coord_num_dir {}".format(self.coord_num_dir))
        else: # center crop
            x_size, y_size = img.width, img.height
            if x_size == self.patch_size:
                xst = 0
            else:
                xst = (x_size - self.patch_size) // 2
            if y_size == self.patch_size:
                yst = 0
            else:
                yst = (y_size - self.patch_size) // 2
            if self.return_ac_coords:
                if self.config.train_params.coord_num_dir == 1:
                    ac_coords = torch.FloatTensor([
                        self.ac_coords_from_ratio(xst / (x_size - self.patch_size), "raw"),
                    ])
                elif self.config.train_params.coord_num_dir == 2:
                    ac_coords = torch.FloatTensor([
                        self.ac_coords_from_ratio(yst / (y_size - self.patch_size), "sin"),
                        self.ac_coords_from_ratio(yst / (y_size - self.patch_size), "cos"),
                    ])
                elif self.config.train_params.coord_num_dir == 4:
                    ac_coords = torch.FloatTensor([
                        self.ac_coords_from_ratio(xst / (x_size - self.patch_size), "sin"),
                        self.ac_coords_from_ratio(xst / (x_size - self.patch_size), "cos"),
                        self.ac_coords_from_ratio(yst / (y_size - self.patch_size), "sin"),
                        self.ac_coords_from_ratio(yst / (y_size - self.patch_size), "cos"),
                    ])
                elif self.config.train_params.coord_num_dir in {3, 21}:
                    ac_coords = torch.FloatTensor([
                        self.ac_coords_from_ratio(xst / (x_size - self.patch_size), "raw"),
                        self.ac_coords_from_ratio(yst / (y_size - self.patch_size), "sin"),
                        self.ac_coords_from_ratio(yst / (y_size - self.patch_size), "cos"),
                    ])
                else:
                    raise ValueError("Unknown coord_num_dir {}".format(self.coord_num_dir))

        cropped = img.crop((yst, xst, yst + self.patch_size, xst + self.patch_size))

        if self.return_ac_coords:
            return cropped, ac_coords
        else:
            return cropped, None


class MultiResolutionDataset(Dataset):
    def __init__(self, split=None, img_dir=None, config=None, is_training=True, 
                 disable_extra_cropping=False, simple_return_full=False, override_full_size=None):

        assert (split is not None) or (img_dir is not None), "Either specify split or img_dir."
        assert (split is None) or (img_dir is None), "Can't specify both split and img_dir at the same time."

        self.split = split
        self.img_dir = img_dir
        self.config = config
        self.resolution = config.train_params.data_size
        self.simple_return_full = simple_return_full


        if self.split is not None:

            if "InOut" in self.config.data_params.dataset:
                self.n_zfill = 5
            else:
                self.n_zfill = 8

            hostname = socket.gethostname()
            cur_lmdb_root = None
            if hostname in LMDB_ROOTS:
                cur_lmdb_root = LMDB_ROOTS[hostname]
                print(" [*] Found lmdb root on local hard drive: {}".format(cur_lmdb_root))
            else:
                for entry in LMDB_ROOTS["unspecified"]:
                    if os.path.exists(entry):
                        print(" [*] Found unspecified lmdb root at {}".format(entry))
                        cur_lmdb_root = entry

            if cur_lmdb_root is None:
                print(" [!] Couldn't find lmdb root on local hard drive, use specification in config file...")
                cur_lmdb_root = config.data_params.lmdb_root

            self.path = os.path.join(cur_lmdb_root, config.data_params.dataset, split)
            if os.path.exists(self.path):
                self.env = lmdb.open(
                    self.path,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
            else:
                raise IOError("Cannot find dataset split {} at {}".format(split, self.path))

            if not self.env:
                raise IOError('Cannot open lmdb dataset', self.path)

            with self.env.begin(write=False) as txn:
                self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
                print(" [*] Loaded data with length {}".format(self.length))


        if self.img_dir is not None:
            self.img_paths = sorted(glob(os.path.join(self.img_dir, "*")))
            self.length = len(self.img_paths)


        if hasattr(config.train_params, "extra_center_crop") and config.train_params.extra_center_crop:
            if disable_extra_cropping:
                extra_center_crop_res = None
            else:
                extra_center_crop_res = config.train_params.extra_center_crop
        else:
            extra_center_crop_res = None

        if hasattr(config.train_params, "extra_pre_resize"):
            pre_resize_op = [MaybeResize(config.train_params.extra_pre_resize)]
        else:
            pre_resize_op = []

        if override_full_size is None:
            raw_resize_size = config.train_params.full_size
        else:
            raw_resize_size = override_full_size

        if is_training: 
            self.transform = transforms.Compose(
                pre_resize_op + [
                    MaybeResize(raw_resize_size),
                    MaybeCenterCrop(extra_center_crop_res), # Center crop for fare comparison
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                pre_resize_op + [
                    MaybeResize(raw_resize_size),
                    MaybeCenterCrop(extra_center_crop_res),
                    # transforms.RandomHorizontalFlip(),
                ]
            )

        self.finalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        if hasattr(config.train_params, "extra_center_crop"):
            if disable_extra_cropping:
                crop_input_size = config.train_params.full_size
            else:
                crop_input_size = config.train_params.extra_center_crop
        else:
            crop_input_size = config.train_params.full_size

        is_styleGAN2_baseline = hasattr(self.config.train_params, "styleGAN2_baseline") and self.config.train_params.styleGAN2_baseline
        is_singan = hasattr(self.config.train_params, "singan") and self.config.train_params.singan
        if is_training:
            assert split=="train", "Unlikely training with testing set or validation set!"
            self.crop_fn = CropPatch(
                input_size=crop_input_size, 
                patch_size=config.train_params.patch_size,
                config=config,
                random_crop=True,
                disable_ac=is_styleGAN2_baseline or is_singan)
        else:
            self.crop_fn = CropPatch(
                input_size=crop_input_size, 
                patch_size=config.train_params.patch_size,
                config=config,
                center_crop=True,
                disable_ac=is_styleGAN2_baseline or is_singan)

        if (not is_training) and hasattr(config.test_params, "calc_fid_ext2") and (config.test_params.calc_fid_ext2):
            self.test_full = True
        else:
            self.test_full = False

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        if self.img_dir is not None:
            full_img = Image.open(self.img_paths[index])
        else:
            try:
                with self.env.begin(write=False) as txn:
                    key = f'{self.resolution}-{str(index).zfill(self.n_zfill)}'.encode('utf-8')
                    img_bytes = txn.get(key)

                buffer = BytesIO(img_bytes)
                if buffer is None:
                    raise ValueError(" [!] Meet empty image while loading with key {}".format(key))
                full_img = Image.open(buffer)
            except Exception as e:
                print(" [!] Error at idx {}".format(index))
                raise e
        full_img = self.transform(full_img)

        ac_coords = None
        if self.simple_return_full:
            data_pack = dict(full=full_img)
        elif self.config.train_params.training_modality == "full":
            data_pack = dict(full=full_img)
        elif self.config.train_params.training_modality == "patch":
            patch, ac_coords = self.crop_fn(full_img)
            data_pack = dict(patch=patch, full=full_img)
        else:
            raise NotImplementedError()

        if self.test_full:
            data_pack["full"] = full_img

        if ac_coords is not None:
            data_pack["ac_coords"] = ac_coords            

        not_img_keys = {"ac_coords"}
        data_pack = {
            k: self.finalize(v) if k not in not_img_keys else v
                for k,v in data_pack.items()}

        return data_pack


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)



if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(config)
        config.var = EasyDict()
    config.var.exp_name = os.path.basename(args.config).split(".")[0]
    print(" [*] Config {} loaded!".format(args.config))

    train_set = MultiResolutionDataset(
        split="train",
        config=config,
        is_training=True)
    valid_set = MultiResolutionDataset(
        split="valid",
        config=config,
        is_training=False) 

    loaders = {
        "train": iter(data.DataLoader(
            train_set,
            batch_size=config.train_params.batch_size,
            sampler=data_sampler(train_set, shuffle=False),
            drop_last=False,
            num_workers=16,
        )),
        "valid": iter(data.DataLoader(
            valid_set,
            batch_size=config.train_params.batch_size,
            sampler=data_sampler(valid_set, shuffle=False),
            drop_last=False,
            num_workers=16
        )),
    }
    #import pdb; pdb.set_trace()

    for i in tqdm(range(len(loaders["train"]))):
        try:
            next(loaders["train"])
        except:
            pass

    for i in tqdm(range(len(loaders["valid"]))):
        try:
            next(loaders["valid"])
        except:
            pass
