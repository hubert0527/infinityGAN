import os
import sys
import pdb
import pickle
import argparse
from glob import glob

import numpy as np
from scipy import linalg
from scipy.stats import entropy
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
from torch import nn
from torch.utils import data
from torchvision import transforms

from libs.calc_inception import load_patched_inception_v3
from torchvision.models.inception import inception_v3


@torch.no_grad()
def extract_fid_feature_from_dir(imgs, inception, batch_size, device):
    nsample = len(imgs)
    n_batch = nsample // batch_size
    resid = nsample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    offset = 0
    for batch in tqdm(batch_sizes):
        if batch == 0:
            continue
        batch_img = imgs[offset : offset+batch]
        if type(batch_img[0]) == str:
            batch_img_tensor = []
            for img in batch_img:
                # read them
                img = Image.open(imgfile).convert("RGB")
                img = transform(img)
                batch_img_tensor.append(img)
            batch_img = torch.stack(batch_img_tensor, dim=0)

        batch_img = batch_img.to(device)
        offset += batch

        # feat = inception(img)[0].view(img.shape[0], -1)
        feat = inception(batch_img.clamp(-1, 1))[0].view(batch_img.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)
    return features


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


def calc_dir_fid(args, imgs1, imgs2, inception, batch_size, device):

    features1 = extract_fid_feature_from_dir(imgs1, inception, batch_size, device)
    features2 = extract_fid_feature_from_dir(imgs2, inception, batch_size, device)

    features1 = features1.numpy()
    features2 = features2.numpy()

    mu1, cov1 = np.mean(features1, 0), np.cov(features1, rowvar=False)
    mu2, cov2 = np.mean(features2, 0), np.cov(features2, rowvar=False)

    fid = calc_fid(mu1, cov1, mu2, cov2, eps=1e-6)
    return fid


def inception_score(imgs, device, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    device -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    upsample = nn.Upsample(size=(299, 299), mode='bilinear').to(device)

    def get_pred(x):
        if resize:
            x = upsample(x)
        x = inception_model(x)
        # return F.softmax(x).data.cpu().numpy()
        return F.softmax(x, dim=-1).data.cpu().numpy()

    n_sample = len(imgs)
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]

    offset = 0
    preds = np.zeros((n_sample, 1000))
    for batch in tqdm(batch_sizes):
        if batch == 0:
            continue
        batch_img = imgs[offset : offset+batch]
        if type(batch_img[0]) == str:

            batch_img_tensor = []
            for img in batch_img:
                # read them
                img = Image.open(imgfile).convert("RGB")
                img = transform(img)
                batch_img_tensor.append(img)

            batch_img = torch.stack(batch_img_tensor, dim=0)

            # import pdb; pdb.set_trace()

        batch_img = batch_img.to(device)

        pred = get_pred(batch_img)
        preds[offset : offset+batch] = pred

        offset += batch
    
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (n_sample // splits): (k+1) * (n_sample // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)



if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=48)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--real-dir', type=str)
    parser.add_argument('--fake-dir', type=str)
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    ## inception
    inception = load_patched_inception_v3().to(device)
    inception.eval()

    if "~" in args.real_dir:
        args.real_dir = os.path.expanduser(args.real_dir)
    if "~" in args.fake_dir:
        args.fake_dir = os.path.expanduser(args.fake_dir)

    """
    Load data
    """
    real_imgs = []
    real_img_paths = sorted(glob(os.path.join(args.real_dir, "*")))
    has_failure = False
    assert len(real_img_paths) > 0
    for ix, imgfile in tqdm(enumerate(real_img_paths), total=len(real_img_paths), desc=f'=> loading imgs from {args.real_dir}'):
        try:
            img = Image.open(imgfile).convert("RGB")
        except Exception as e:
            print(ix, e)
            has_failure = True
            continue
        img = transform(img)
        real_imgs.append(img)
    real_imgs = torch.stack(real_imgs, dim=0)

    real_h, real_w = real_imgs[0, 0].shape

    fake_imgs = []
    fake_img_paths = sorted(glob(os.path.join(args.fake_dir, "*")))
    assert len(fake_img_paths) > 0
    for ix, imgfile in tqdm(enumerate(fake_img_paths), total=len(fake_img_paths), desc=f'=> loading imgs from {args.fake_dir}'):
        try:
            img = Image.open(imgfile).convert("RGB")
        except Exception as e:
            print(ix, e)
            has_failure = True
            continue
        img = transform(img)

        # Center crop, since InfinityGAN will generate slightly larger outpainting results
        _, fake_h, fake_w = img.shape
        pad_h = (fake_h - real_h)
        pad_w = (fake_w - real_w)
        img = img[:, pad_h:pad_h+real_h, pad_w:pad_w+real_w]

        fake_imgs.append(img)
    fake_imgs = torch.stack(fake_imgs, dim=0)

    assert (not has_failure), "Receives error(s) while loading data, please check the messages above!"

    ## IS
    # is_score_real_mean, is_score_real_std = \
    #     inception_score(real_imgs, device, batch_size=args.batch, resize=True, splits=1)
    is_score_fake_mean, is_score_fake_std = \
        inception_score(fake_imgs, device, batch_size=args.batch, resize=True, splits=1)
    # print(f'=> {args.real_dir} IS: {is_score_real}')
    print('=> IS: {:.6f} +- {:.6f}'.format(is_score_fake_mean, is_score_fake_std))

    ## FID
    fid = calc_dir_fid(args, real_imgs, fake_imgs, inception, args.batch, device)
    print('=> FID: {:.6f}'.format(fid))

    # Generate report
    def fprint(f, s):
        print(s)
        f.write(s+"\n")
    exp_toks = [tok for tok in args.fake_dir.split("/")]
    exp_name = exp_toks[exp_toks.index("logs") + 1]
    log_file = os.path.join("./logs-quant/outpaint/", exp_name+".txt")
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    with open(log_file, "a") as f:
        fprint(f, "*" * 30)
        fprint(f, " [*] EXP: {}".format(exp_name))
        fprint(f, " [*] PATH: {}".format(args.fake_dir))
        fprint(f, '  => IS: {:.6f} +- {:.6f}'.format(is_score_fake_mean, is_score_fake_std))
        fprint(f, '  => FID: {:.6f}'.format(fid))
        fprint(f, "*" * 30)
