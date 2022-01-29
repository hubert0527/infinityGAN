import os
import argparse
import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pickle as pkl
from scipy import linalg
from tqdm import tqdm

from libs.calc_inception import load_patched_inception_v3


def gan_img_to_classify_img(img):
    img = (img + 1) / 2
    return torch.clamp(img, 0, 1)

def spatial_partition_cat_func(img, inception, num_patches=16):
    B, C, H, W = img.shape
    patch_len = int(np.sqrt(num_patches))
    assert patch_len**2 == num_patches

    patch_size = img.shape[2] // patch_len
    #assert img.shape[2] % patch_len == 0, "Got image shape {}, cannot be partitioned in {} patches".format(img.shape, patch_len)
    assert img.shape[2] == img.shape[3]

    patches = []
    for i in range(patch_len):
        for j in range(patch_len):
            patch = img[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch)
    patches = torch.cat(patches, 0)
    patches_feat = inception(patches)[0].squeeze()
    inception_ch = patches_feat.shape[-1]
    patches_feat = patches_feat.view(B, num_patches, inception_ch).view(B, num_patches*inception_ch)
    return patches_feat
            

@torch.no_grad()
def extract_feature_from_samples(
    generation_fn, inception, batch_size, n_sample, device, 
    cocogan_patched_fid = False, create_graph_only=False, spatial_partition_cat=False, assert_eval_shape=None):

    if cocogan_patched_fid:
        n_sample = n_sample // 16 + 1 # hard coded
    
    if create_graph_only:
        n_batch = 1
    else:
        if n_sample % batch_size == 0:
            n_batch = n_sample // batch_size 
        else:
            n_batch = n_sample // batch_size + 1
    features = []

    for img in tqdm(generation_fn(n_batch), total=n_batch):
        if isinstance(inception, nn.DataParallel): # Dataparallel takes CPU tensor
            img = img.cpu()
        elif next(inception.parameters()).is_cuda:
            img = img.cuda()
        else:
            img = img.cpu()
        img = gan_img_to_classify_img(img)
        if img.shape[1] == 1: # grayscale
            img = img.repeat(1, 3, 1, 1)

        if assert_eval_shape is not None:
            assert (img.shape[2] == assert_eval_shape) and (img.shape[3] == assert_eval_shape), \
                "Set assert image shape {}, but got {}".format(assert_eval_shape, img.shape)

        if cocogan_patched_fid:
            patches = []
            patch_size = img.shape[2] // 4
            for i in range(4):
                for j in range(4):
                    xst = i*patch_size
                    yst = j*patch_size
                    xed = xst + patch_size
                    yed = yst + patch_size
                    patches.append(img[:, :, xst:xed, yst:yed])
            img = torch.cat(patches, 0)

        if spatial_partition_cat:
            feat = spatial_partition_cat_func(img, inception, num_patches=16)
        else:
            feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)

    return features[:n_sample]


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid

@torch.no_grad()
def calc_stats_from_dataloader(dataloader, inception, config, n_samples, modality, 
                               cocogan_patched_fid=False, spatial_partition_cat=False, assert_eval_shape=None):
    features = []
    total = 0
    iterator = iter(dataloader)
    batch_size = config.train_params.batch_size
    n_batches = int(np.ceil(n_samples / batch_size))
    if cocogan_patched_fid:
        n_batches = n_batches // 16 + 1
    for _ in tqdm(range(n_batches)):
        real_data = next(iterator)
        if modality is None:
            img = real_data
        else:
            img = real_data[modality]
        if img.min() > 0:
            print(" [!] Whole batch has minimum intensity > 0, can be a bug if happens to all batches! (in `libs.fid.calc_stats_from_dataloader`)")
        img = gan_img_to_classify_img(img)
        if (not isinstance(inception, nn.DataParallel)): # Dataparallel takes CPU tensor
            if next(inception.parameters()).is_cuda:
                img = img.cuda()
        if img.shape[1] == 1: # grayscale
            img = img.repeat(1, 3, 1, 1)

        if assert_eval_shape is not None:
            assert (img.shape[2] == assert_eval_shape) and (img.shape[3] == assert_eval_shape), \
                "Set assert image shape {}, but got {}".format(assert_eval_shape, img.shape)

        if cocogan_patched_fid:
            patches = []
            patch_size = img.shape[2] // 4
            for i in range(4):
                for j in range(4):
                    xst = i*patch_size
                    yst = j*patch_size
                    xed = xst + patch_size
                    yed = yst + patch_size
                    patches.append(img[:, :, xst:xed, yst:yed])
            img = torch.cat(patches, 0)

        if spatial_partition_cat:
            feat = spatial_partition_cat_func(img, inception, num_patches=16)
        else:
            feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))
        total += img.shape[0]
        if total > n_samples:
            break

    features = torch.cat(features, 0).numpy()[:n_samples]
    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)
    return sample_mean, sample_cov


def eval_fid(dataloader, generation_fn, inception, dataset_key, modality, device, config, 
             no_write_cache=False, create_graph_only=False, spatial_partition_cat=False, cocogan_patched_fid=False,
             assert_eval_shape=None, external_cache_root=None):
    if external_cache_root is None:
        cache_root = ".fid-cache/"
    else:
        cache_root = external_cache_root
        assert os.path.exists(cache_root), \
            "Specified a non-existing fid cache root at {}".format(external_cache_root)

    if not os.path.exists(cache_root): os.makedirs(cache_root)
    cache_path = os.path.join(cache_root, "{}.pkl".format(dataset_key))
    real_stats = None
    if os.path.exists(cache_path):
        try:
            real_stats = pkl.load(open(cache_path, "rb"))
        except:
            real_stats = None
    if real_stats is not None:
        real_mean = real_stats['mean']
        real_cov = real_stats['cov']
    elif create_graph_only:
        real_mean, real_cov = None, None
    else:
        print(" [!] Failed to find FID stats of dataset_key={}, start generating cache...".format(dataset_key))
        real_mean, real_cov = calc_stats_from_dataloader(
            dataloader, inception, config, config.test_params.n_fid_sample, modality=modality, 
            cocogan_patched_fid=cocogan_patched_fid,
            spatial_partition_cat=spatial_partition_cat, assert_eval_shape=assert_eval_shape)
        if (not no_write_cache):
            pkl.dump({"mean": real_mean, "cov": real_cov}, open(cache_path, "wb"), protocol=4)

    if isinstance(generation_fn, DataLoader): # Eval with image folder instead of generator
        sample_mean, sample_cov = calc_stats_from_dataloader(
            generation_fn, inception, config, config.test_params.n_fid_sample, modality=modality, 
            cocogan_patched_fid=cocogan_patched_fid,
            spatial_partition_cat=spatial_partition_cat, assert_eval_shape=assert_eval_shape)
        if create_graph_only:
            return 500
    else:
        features = extract_feature_from_samples(
            generation_fn, 
            inception, 
            config.train_params.batch_size, 
            config.test_params.n_fid_sample, 
            device,
            cocogan_patched_fid=cocogan_patched_fid,
            spatial_partition_cat=spatial_partition_cat,
            create_graph_only=create_graph_only,
            assert_eval_shape=assert_eval_shape,
        ).numpy()

        if create_graph_only:
            return 500

        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)

    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

    return fid

