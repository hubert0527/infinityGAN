import math

import torch
from torch import nn, autograd
from torch.nn import functional as F



def l1_loss(a, b, reduce_all=False):
    if reduce_all:
        return (a-b).abs().mean()
    else: # return in batch shape
        bs = a.shape[0]
        return (a - b).abs().view(bs, -1).mean(1)


def l2_loss(a, b, reduce_all=False):
    if reduce_all:
        return 0.5 * ((a-b)**2).mean()
    else: # return in batch shape
        bs = a.shape[0]
        return (0.5 * (a - b)**2).view(bs, -1).mean(1)


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_logistic_loss_fake(fake_pred):
    fake_loss = F.softplus(fake_pred)
    return fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def grad_reduce(grad):
    if grad.ndim == 2:
        return torch.sqrt(grad.pow(2).mean(1))
    elif grad.ndim == 3:
        return torch.sqrt(grad.pow(2).mean([1,2]))
    elif grad.ndim == 4:
        return torch.sqrt(grad.pow(2).mean([1,2,3]))
    else:
        raise NotImplementedError()


def calc_path_lengths(fake_img, latents):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grads = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = [grad_reduce(grad) for grad in grads]
    return path_lengths


def g_path_regularize(path_lengths, mean_path_lengths, decay=0.01):
    path_penalty = 0
    new_path_means = []
    for path_length,mean_path_length in zip(path_lengths, mean_path_lengths):
        path_mean = mean_path_length.item() + decay * (path_length.mean() - mean_path_length)
        path_penalty += (path_length - path_mean).pow(2).mean()
        new_path_means.append(path_mean.detach())
    return path_penalty, new_path_means
    

def coord_ac_loss(pred, label, side, config):
    if label.device != pred.device:
        label = label.to(pred.device) # Workaround for cpu device of Dataparallel

    if hasattr(config.train_params, "coord_ac_vert_only") and config.train_params.coord_ac_vert_only:
        return l1_loss(pred[:, 0], label[:, 0], reduce_all=True)
    else:
        if hasattr(config.train_params, "coord_ac_categorical") and config.train_params.coord_ac_categorical:
            assert config.train_params.coord_ac_vert_only, "experimental setup"
            label = ((label[:, 0] + 1) * 2 * config.train_params.coord_vert_sample_size).uint8() # [-1, 1] => class
            return F.cross_entropy(pred, label)
        else:
            return l1_loss(pred, label, reduce_all=True)


def noise_regularize(noises):
    loss = 0
    for noise in noises:
        while True:
            _, _, size_h, size_w = noise.shape
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )
            if min(size_h, size_w) <= 8:
                break
            if size_h % 2 != 0:
                noise = noise[:, :, :-1, :]
            if size_w % 2 != 0:
                noise = noise[:, :, :, :-1]
            noise = noise.reshape([-1, 1, size_h // 2, 2, size_w // 2, 2])
            noise = noise.mean([3, 5])
    return loss

