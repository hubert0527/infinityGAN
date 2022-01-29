import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import entropy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from torchvision.models.inception import inception_v3


def inception_score(imgs, device, batch_size=32, resize=False, splits=1, transform=None):
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
                img = Image.open(img).convert("RGB")
                img = transform(img)
                batch_img_tensor.append(img)
            batch_img = torch.stack(batch_img_tensor, dim=0)

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