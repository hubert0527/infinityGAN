import os
import numpy as np
import torch
from torch.functional import align_tensors
import torch.nn.functional as F
from glob import glob
from easydict import EasyDict
from torch.utils import data
from torch.utils.data import Dataset, BatchSampler, SequentialSampler
from torchvision import transforms
import torchvision.transforms.functional as ttF
from PIL import Image

class QuantEvalSampleGenerator():
    def __init__(self, g_ema, latent_sampler, output_size, use_seq_inf, postprocessing_params, device, config, fid_type,
                 img_folder=None, sample_fn=None, use_pil_resize=False):
        self.g_ema = g_ema
        self.latent_sampler = latent_sampler
        self.output_size = output_size
        self.use_seq_inf = use_seq_inf
        self.postprocessing_params = postprocessing_params
        self.device = device
        self.config = config
        self.fid_type = fid_type
        self.use_pil_resize = use_pil_resize

        self.use_img_folder = (img_folder is not None)
        self.use_sample_fn = (sample_fn is not None)

        assert isinstance(postprocessing_params, list), \
            "Postprocessing params should be passed in with an ordered list."

        # Procedures that use external sources
        if img_folder is not None:
            assert False, "Not tested."
            dataset = ImageFolder(img_folder)
            self.img_folder_loader = DataLoader(
                dataset=dataset,
                batch_sampler=BatchSampler(
                    SequentialSampler(dataset), batch_size=self.config.train_params.batch_size, drop_last=False),
                num_workers=8)
        elif sample_fn is not None:
            self.sample_fn = sample_fn
        # Procedures that requires generate images
        elif hasattr(config.train_params, "styleGAN2_baseline") and config.train_params.styleGAN2_baseline:
            mult = int(np.ceil(output_size / config.train_params.patch_size))
            self.latent_size = config.train_params.ts_input_size * mult
        else: # InfinityGAN generation at testing time
            if fid_type == "alis":
                from test_managers.fused_seq_connecting_generation import FusedSeqConnectingGenerationManager
                self.task_manager = FusedSeqConnectingGenerationManager(g_ema, device, None, config)
                self.task_manager.task_specific_init()
            elif self.use_seq_inf:
                from test_managers.infinite_generation import InfiniteGenerationManager
                config.task = EasyDict()
                self.task_manager = InfiniteGenerationManager(g_ema, device, None, config)
                self.task_manager.task_specific_init(output_size=(output_size, output_size))
            else:
                self.task_manager = None
                g_ema_module = g_ema.module if config.var.dataparallel else g_ema
                self.latent_size = g_ema_module.calc_in_spatial_size(output_size)

    def run_postprocessing(self, images):
        for name,param in self.postprocessing_params:
            if name == "crop":
                images = self._center_crop(images, param)
            elif name == "scale":
                assert param < 10, \
                    "Scaling function get an unexpected large value {}".format(param) +\
                    " are you accidentally feeding target resolution instead of a scale?"
                images = self._scale(images, param)
            elif name == "resize":
                assert param > 1, \
                    "Scaling function get an unexpected small value {}".format(param) +\
                    " are you accidentally feeding scale instead of a target resolution?"
                images = self._resize(images, param)
            elif name == "assert":
                self._assert(images, param)
            elif name == "img_to_gan_img":
                images = self._img_to_gan_img(images)
            elif name == "np_to_tensor":
                images = self._np_to_tensor(images)
            else:
                raise NotImplementedError(
                    "Unkown postprocessing method {} with param {}".format(name, param))
            # print(" Shape after {} ({}) is {}".format(name, param, images.shape))
        return images

    def _gan_tensor_to_pil_img(self, images):
        images = (images + 1) / 2
        images = images.cpu()
        return [ttF.to_pil_image(im) for im in images]

    def _pil_img_to_gan_tensor(self, images):
        images = torch.stack([ttF.to_tensor(im) for im in images])
        return (images * 2) - 1

    def _scale(self, images, scale):
        #if self.use_pil_resize:
        #    h, w = images[0].height, images[0].width
        #    new_h = round(h * scale)
        #    new_w = round(w * scale)
        #    return [im.resize([new_w, new_h], resample=Image.BILINEAR) for im in images]
        #else:
        return F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=True)

    def _resize(self, images, target_size):
        if self.use_pil_resize:
            device = images.device
            images = self._gan_tensor_to_pil_img(images)
            images = [im.resize([target_size, target_size], resample=Image.BILINEAR) for im in images]
            images = self._pil_img_to_gan_tensor(images)
            return images.to(device)
        else:
            return F.interpolate(images, size=target_size, mode="bilinear", align_corners=True)

    def _assert(self, images, target_shape):
        assert images.shape[-1] == target_shape, \
            "Assert shape to be {}, but get {}!".format(target_shape, images.shape[-1]) 
        assert images.shape[-2] == target_shape, \
            "Assert shape to be {}, but get {}!".format(target_shape, images.shape[-2]) 

    def _img_to_gan_img(self, images):
        assert (images.min() > 0) and (images.max() < 1), \
            "Got images not in [0, 1] range, with range ({}, {})".format(images.min(), images.max())
        return images * 2 - 1

    def _np_to_tensor(self, images):
        return torch.from_numpy(images).permute(0, 3, 1, 2).to(self.device)

    def _center_crop(self, images, size):
        _, _, H, W = images.shape
        if H <= size:
            assert H == W, \
                "Images here should all be squared, but got {}".format(images.shape)
            return images
        pad_h = (H - size) // 2
        pad_w = (W - size) // 2
        return images[:, :, pad_h:pad_h+size, pad_w:pad_w+size]

    @torch.no_grad()
    def __call__(self, n_batch):

        for _ in range(n_batch):

            if self.use_img_folder:
                images = next(self.img_folder_loader)
                images = images.to(self.device)
            elif self.use_sample_fn:
                images = self.sample_fn()
            elif self.use_seq_inf or self.fid_type=="alis":
                images = self.task_manager.run_next(save=False, write_gpu_time=False, disable_pbar=True)
            else:
                global_latent = self.latent_sampler.sample_global_latent(
                    self.config.train_params.batch_size, self.device)
                local_latent = self.latent_sampler.sample_local_latent(
                    self.config.train_params.batch_size, self.device, 
                    specific_shape=(self.latent_size, self.latent_size))
        
                sample = self.g_ema(
                    global_latent=global_latent, 
                    local_latent=local_latent,
                    disable_dual_latents=True)
                images = sample["gen"]
            images = self.run_postprocessing(images)
            yield images.contiguous()


class QuantEvalDataLoader():
    def __init__(self, dataset, output_size, device, config):
        self.dataset = dataset
        self.output_size = output_size
        self.device = device

        if hasattr(config, "task") and hasattr(config.task, "init_index"):
            from utils import data_sampler # Avoid import error from COCO-GAN
            sampler = data_sampler(
                dataset, shuffle=False, init_index=config.task.init_index)
        else:
            sampler = data.SequentialSampler(dataset)

        self.dataloader_proto = data.DataLoader(
            dataset,
            batch_size=config.train_params.batch_size,
            sampler=sampler,
            drop_last=False,
            num_workers=16,
        )
        self.dataloader = None

    def _center_crop(self, images):
        _, _, H, W = images.shape
        if H == self.output_size and W == self.output_size:
            return images
        else:
            assert self.output_size < H, "Got {} > {}".format(self.output_size, H)
            assert self.output_size < W, "Got {} > {}".format(self.output_size, W)
        pad_h = (H - self.output_size) // 2
        pad_w = (W - self.output_size) // 2
        return images[:, :, pad_h:pad_h+self.output_size, pad_w:pad_w+self.output_size]

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        images = next(self.dataloader)["full"]
        assert images.shape[2] == self.output_size and images.shape[3] == self.output_size, \
            "Got unexpected image shape {} instead of size {}".format(images.shape, self.output_size)
        return images
        # return self._center_crop(images["full"])

    def __iter__(self):
        self.dataloader = iter(self.dataloader_proto)
        return self


class ImageFolder(Dataset):
    # torchvision's requires labels...
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_paths = glob(os.path.join(self.image_dir, "*"))
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = Image.open(image_path)
        return self.transforms(img)
    
    def __len__(self):
        return len(self.image_paths)
