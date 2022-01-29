# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F


def DiffAugmentDual(a, b, policy='', channels_first=True):
    assert a.shape == b.shape, "Got {} != {}".format(a.shape, b.shape)
    if policy:
        if not channels_first:
            a = a.permute(0, 3, 1, 2)
            b = b.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                a, b = f(a, b)
        if not channels_first:
            a = a.permute(0, 2, 3, 1)
            b = b.permute(0, 2, 3, 1)
        a = a.contiguous()
        b = b.contiguous()
    return a, b


def rand_brightness(a, b):
    aug = (torch.rand(a.size(0), 1, 1, 1, dtype=a.dtype, device=a.device) - 0.5)
    a = a + aug
    b = b + aug 
    return a, b


def rand_saturation(a, b):
    a_mean = a.mean(dim=1, keepdim=True)
    b_mean = b.mean(dim=1, keepdim=True)
    aug = (torch.rand(a.size(0), 1, 1, 1, dtype=a.dtype, device=a.device) * 2)
    a = (a - a_mean) * aug + a_mean
    b = (b - b_mean) * aug + b_mean
    return a, b


def rand_contrast(a, b):
    a_mean = a.mean(dim=[1, 2, 3], keepdim=True)
    b_mean = b.mean(dim=[1, 2, 3], keepdim=True)
    aug = (torch.rand(a.size(0), 1, 1, 1, dtype=a.dtype, device=a.device) + 0.5)
    a = (a - a_mean) * aug + a_mean
    b = (b - b_mean) * aug + b_mean
    return a, b


def rand_translation(a, b, ratio=0.125):
    shift_x, shift_y = int(a.size(2) * ratio + 0.5), int(a.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[a.size(0), 1, 1], device=a.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[a.size(0), 1, 1], device=a.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(a.size(0), dtype=torch.long, device=a.device),
        torch.arange(a.size(2), dtype=torch.long, device=a.device),
        torch.arange(a.size(3), dtype=torch.long, device=a.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, a.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, a.size(3) + 1)

    a_pad = F.pad(a, [1, 1, 1, 1, 0, 0, 0, 0])
    b_pad = F.pad(b, [1, 1, 1, 1, 0, 0, 0, 0])
    a = a_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    b = b_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    # print('translation!!!')
    return a, b


def rand_cutout(a, b, ratio=0.5):
    cutout_size = int(a.size(2) * ratio + 0.5), int(a.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, a.size(2) + (1 - cutout_size[0] % 2), size=[a.size(0), 1, 1], device=a.device)
    offset_y = torch.randint(0, a.size(3) + (1 - cutout_size[1] % 2), size=[a.size(0), 1, 1], device=a.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(a.size(0), dtype=torch.long, device=a.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=a.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=a.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=a.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=a.size(3) - 1)
    mask = torch.ones(a.size(0), a.size(2), a.size(3), dtype=a.dtype, device=a.device)
    mask[grid_batch, grid_x, grid_y] = 0
    a = a * mask.unsqueeze(1)
    b = b * mask.unsqueeze(1)
    # print('cutout!!!')
    return a, b


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}