import os
import random
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.autograd import Function, Variable

from coord_handler import CoordHandler
from models.ops import *
from models.losses import calc_path_lengths
from dataset import DictTensor



def sequential_get_flops(sequential_module, inputs):
    flops = 0
    h = inputs
    for layer in sequential_module:
        flops += layer.get_flops(h)
        h = sequential_module(h)
    return h, flops
    
def create_fusion_styles(cur_fusion_map, styles, config):
    """
    (N: number of styles)
    cur_fusion_map: shape (B, N, H, W)
    styles        : [shape (B, C)] * N
    """
    device = styles[0].device
    bs, _, cur_height, cur_width = cur_fusion_map.shape
    fused_style = torch.zeros(
        bs, 
        config.train_params.global_latent_dim,
        cur_height,
        cur_width)
    fused_style = fused_style.to(device)

    for i in range(len(config.task.style_centers)):
        fused_style += cur_fusion_map[:, i:i+1].to(device) * styles[i].unsqueeze(2).unsqueeze(3)
    return fused_style


def setup_paired_inputs(inputs):
    # Make a batch of samples, (e.g., [A, B, C, D]) into paired form (e.g., [A, A, C, C])
    #
    # [Warning] I'm not sure if this resulting in inplace replacement. As a result, this
    # function should only be used in input nodes. Do not apply it on intermediate data such
    # as z_spatial.
    inputs_clone = inputs.detach().clone()

    batch_size = inputs_clone.shape[0]
    if batch_size%2 == 0: # Easy case
        inputs_clone[1::2] = inputs_clone[0::2]
    else:
        inputs_clone[1::2] = inputs_clone[0:-1:2]
    return inputs_clone


def flatten(v):
    sp_dim = np.prod(v.shape[1:])
    return v.reshape(v.shape[0], sp_dim) if v.ndim > 2 else v


def angular_similarity(a, b):
    a, b = flatten(a), flatten(b)
    # Some how detaching this in high-res setup will make model explode
    denom = (a.norm(2, dim=1) * b.norm(2, dim=1)) #.detach()
    cosine_sim = (a * b).sum(1) / denom
    return (1 - torch.acos(cosine_sim) / np.pi)


class ConditionalBlock(nn.Module):
    def __init__(self, idx, config):
        super().__init__()
        self.config = config
        self.ss_disable_noise = config.train_params.ss_disable_noise

        local_dim = config.train_params.local_latent_dim
        global_dim = config.train_params.global_latent_dim

        in_channel = local_dim
        if config.train_params.ss_coord_all_layers:
            in_channel += self.config.train_params.coord_num_dir
        elif idx==0:
            in_channel += self.config.train_params.coord_num_dir

        kernel_size = config.train_params.ss_unfold_radius * 2 + 1
        self.conv = StyledConv(
            in_channel=in_channel,
            out_channel=local_dim,
            kernel_size=kernel_size,
            style_dim=global_dim,
            no_zero_pad=True,
            disable_noise=self.ss_disable_noise,
            config=config,
            side="ss")
    
    def calibrate_spatial_shape(self, feature, direction, padding_mode="replicate", verbose=False, pin_loc=None):
        return self.conv.calibrate_spatial_shape(feature, direction, padding_mode=padding_mode, verbose=verbose, pin_loc=pin_loc)

    def forward(self, x, cond, coords, noise=None, test_ids=None, calc_flops=False):
        if self.config.train_params.ss_coord_all_layers:
            x = torch.cat([x, coords], 1)
        else:
            pass # Already assigned coord at the beginning
        return self.conv(x, cond, noise=noise, coords=coords, test_ids=test_ids, calc_flops=calc_flops)


class ImplicitFunction(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_res_block = config.train_params.ss_n_layers

        convs = [
            ConditionalBlock(idx=i, config=config) 
                for i in range(num_res_block)
        ]

        self.conv_stack = nn.Sequential(*convs)

        if hasattr(config.train_params, "ss_mapping") and config.train_params.ss_mapping:
            n_mlp = 8
            global_latent_dim = config.train_params.global_latent_dim
            lr_mlp = 0.01
            layers = [PixelNorm()]
            for i in range(n_mlp):
                layers.append(
                    EqualLinear(
                        global_latent_dim, global_latent_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                    )
                )
            self.global_mapping = nn.Sequential(*layers)
        else:
            self.global_mapping = None

    def _select_center(self, src, ref):
        pad_h = (src.shape[2] - ref.shape[2]) // 2
        pad_w = (src.shape[3] - ref.shape[3]) // 2
        if pad_h == 0 and pad_w == 0:
            return src
        fh, fw = ref.shape[2], ref.shape[3]
        return src[:, :, pad_h:pad_h+fh, pad_w:pad_w+fw]

    def calibrate_spatial_shape(self, feature, direction, padding_mode="replicate", verbose=False, pin_loc=None):
        # We only calibrates for the convs in the main stream, 
        # the to_rgb stream can directly use the center-cropped region.
        ret_latents = []
        ret_pin_locs = []

        if direction == "forward":
            ops = self.conv_stack
        elif direction == "backward":
            ops = self.conv_stack[::-1]
        else:
            raise NotImplementedError()

        for conv in ops:
            feature, pin_loc = \
                conv.calibrate_spatial_shape(feature, direction, padding_mode=padding_mode, verbose=verbose, pin_loc=pin_loc)
            ret_latents.append(feature)
            ret_pin_locs.append(pin_loc)

        return ret_latents, ret_pin_locs

    def forward(self, global_latent, local_latent, coords, noises=None, test_ids=None, calc_flops=False):

        if self.global_mapping is not None:
            global_latent = self.global_mapping(global_latent)

        h = local_latent
        cond = global_latent
        flops = 0
        for i,conv in enumerate(self.conv_stack):
            coords = self._select_center(src=coords, ref=h)
            if isinstance(cond, list):
                cur_cond = cond[i]
            else:
                cur_cond = cond
            if noises is not None:
                if isinstance(cond, list):
                    cur_noise = noises[i]
                else:
                    cur_noise = noises
            else:
                cur_noise = None

            h, cur_flops = conv(h, cur_cond, coords, noise=cur_noise, test_ids=test_ids, calc_flops=calc_flops)
            flops += cur_flops

        return h, flops


class StructureSynthesizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.coord_handler = CoordHandler(config)
        self.implicit_model = ImplicitFunction(config)

        if hasattr(self.config.train_params, "diversity_z_w") and self.config.train_params.diversity_z_w!=0:
            self.use_div_z_loss = True
        else:
            self.use_div_z_loss = False

        if hasattr(self.config.train_params, "diversity_angular") and self.config.train_params.diversity_angular != 0:
            self.use_angular_div = True
        else:
            self.use_angular_div = False


    def calc_out_spatial_size(self, in_spatial_size, return_list=False):
        if return_list:
            assert False, "Unused old codes, use with care!"
            # ret = []
            # single_ss_unfold_size = self.config.train_params.ss_unfold_radius
            # cur_size = in_spatial_size
            # for i in range(self.config.train_params.ss_n_layers):
            #     next_size = cur_size - single_ss_unfold_size * 2
            #     if i == (self.config.train_params.ss_n_layers-1): 
            #         next_size -= 2
            #     ret.append(next_size)
            #     cur_size = next_size
            # return ret
        else:
            ss_unfold_size = self.config.train_params.ss_n_layers * self.config.train_params.ss_unfold_radius
            return in_spatial_size - ss_unfold_size * 2
        
    def calibrate_spatial_shape(self, feature, direction, padding_mode="replicate", verbose=False, pin_loc=None):
        return self.implicit_model.calibrate_spatial_shape(feature, direction, padding_mode=padding_mode, verbose=verbose, pin_loc=pin_loc)

    def _diversity_latent_dist(self, global_latent, local_latent):
        if self.use_angular_div:
            if local_latent.shape[0]%2 == 0:
                z_dist = angular_similarity(local_latent[0::2], local_latent[1::2]).mean()
            else:
                z_dist = angular_similarity(local_latent[0:-1:2], local_latent[1::2]).mean()
        else:
            if local_latent.shape[0]%2 == 0:
                z_dist = (local_latent[0::2] - local_latent[1::2]).abs().mean()
            else:
                z_dist = (local_latent[0:-1:2] - local_latent[1::2]).abs().mean()
        return z_dist

    def _diversity_image_dist(self, syn_feat):
        if self.use_angular_div:
            if syn_feat.shape[0]%2 == 0:
                x_dist = angular_similarity(syn_feat[0::2], syn_feat[1::2]).mean()
            else:
                x_dist = angular_similarity(syn_feat[0:-1:2], syn_feat[1::2]).mean()
        else:
            if syn_feat.shape[0]%2 == 0:
                x_dist = (syn_feat[0::2] - syn_feat[1::2]).abs().mean()
            else:
                x_dist = (syn_feat[0:-1:2] - syn_feat[1::2]).abs().mean()
        return x_dist

    def diversity_z_loss(self, global_latent, local_latent, syn_feat, eps=1e-5):
        z_dist = self._diversity_latent_dist(global_latent, local_latent)
        x_dist = self._diversity_image_dist(syn_feat)
        div_loss = 1 / (x_dist/z_dist + eps)
        return div_loss

    def get_coords(self, local_latent, override_coords=None, is_fid_eval=False, disable_dual_latents=False):
        coords, ac_coords = self.coord_handler.sample_coord_grid(
            local_latent, 
            is_training=self.training,
            is_fid_eval=is_fid_eval,
            override_coords=override_coords,
            return_ac_coords=True)

        if self.use_div_z_loss:
            if disable_dual_latents:
                pass
            else:
                assert self.training
                coords = setup_paired_inputs(coords)
                
        return coords, ac_coords

    def forward(
        self, 

        # Possible inputs
        global_latent, 
        local_latent, 
        override_coords=None, 
        noises=None, 
        style_fusion_map=None,
        test_ids=None,

        # Control signals
        disable_dual_latents=False, 
        is_fid_eval=False,
        calc_flops=False):

        flops = 0

        coords, ac_coords = self.get_coords(
            local_latent, override_coords=override_coords, 
            is_fid_eval=is_fid_eval, disable_dual_latents=disable_dual_latents)

        if style_fusion_map is not None:
            # global_latent: [Tensor (BxC)] * N => Tensor (BxCxHxW)
            # Note: The input global_latent is already a single (non-style-fused) latent here
            global_latent = create_fusion_styles(style_fusion_map, global_latent, self.config)

        """
        GNN Prop
        """
        if self.config.train_params.ss_coord_all_layers:
            pass # Assign coordinates within each conv
        else:
            local_latent = torch.cat([
                local_latent,
                coords
            ], 1)

        output, cur_flops = self.implicit_model(
            global_latent, local_latent, coords=coords, noises=noises, test_ids=test_ids, calc_flops=calc_flops)
        flops += cur_flops

        return output, coords, ac_coords, flops

class TextureSynthesizer(nn.Module):
    def __init__(self, config):
        super().__init__()

        is_styleGAN2_baseline = (hasattr(config.train_params, "styleGAN2_baseline") and config.train_params.styleGAN2_baseline)

        self.config = config
        if config.train_params.training_modality == "patch":
            self.size = config.train_params.patch_size
        else:
            raise NotImplementedError()
        self.global_latent_dim = config.train_params.global_latent_dim
        self.local_latent_dim = config.train_params.local_latent_dim
        n_mlp = config.train_params.n_mlp
        channel_multiplier = config.train_params.channel_multiplier

        if config.train_params.ts_no_zero_pad:
            blur_kernel = [1, 2, 1]
        else:
            assert hasattr(self.config.train_params, "styleGAN2_baseline") and self.config.train_params.styleGAN2_baseline
            blur_kernel = [1, 3, 3, 1]

        lr_mlp = 0.01
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    self.global_latent_dim, self.global_latent_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.mapping = nn.Sequential(*layers)

        # self.channels = {
        #     4: self.local_latent_dim,
        #     8: 512,
        #     16: 512,
        #     32: 512,
        #     64: 256 * channel_multiplier,
        #     128: 128 * channel_multiplier,
        #     256: 64 * channel_multiplier,
        #     512: 32 * channel_multiplier,
        #     1024: 16 * channel_multiplier,
        # }

        if config.train_params.training_modality == "patch":
            g_output_res = config.train_params.patch_size
        elif config.train_params.training_modality == "full":
            g_output_res = config.train_params.full_size
        else:
            raise NotImplementedError()

        if g_output_res == 101 and config.train_params.ts_input_size == 11:
            self.convs_specs = [
                dict(out_ch=512, upsample=True), # 11 -> 19
                # skip-node 0
                dict(out_ch=512, upsample=False), # 19 -> 17
                # skip-node 1
                dict(out_ch=512, upsample=True), # 17 -> 31
                # skip-node 2
                dict(out_ch=512, upsample=False), # 31 -> 29
                # skip-node 3
                dict(out_ch=512, upsample=True), # 29 -> 55
                # skip-node 4
                dict(out_ch=512, upsample=False), # 55 -> 53
                # skip-node 5
                dict(out_ch=256*channel_multiplier, upsample=True), # 53 -> 103
                # skip-node 6
                dict(out_ch=256*channel_multiplier, upsample=False), # 103 -> 101
                # skip-node 7
            ]
            self.to_rgbs_specs = [
                dict(src=1, tgt=3,  upsample=True),
                dict(src=3, tgt=5,  upsample=True),
                dict(src=5, tgt=7,  upsample=True),
                dict(src=7, tgt=8,  upsample=True),
            ]
        elif g_output_res == 197 and config.train_params.ts_input_size == 11:
            self.convs_specs = [
                dict(out_ch=512, upsample=True), # 11 -> 19
                # skip-node 0
                dict(out_ch=512, upsample=False), # 19 -> 17
                # skip-node 1
                dict(out_ch=512, upsample=True), # 17 -> 31
                # skip-node 2
                dict(out_ch=512, upsample=False), # 31 -> 29
                # skip-node 3
                dict(out_ch=512, upsample=True), # 29 -> 55
                # skip-node 4
                dict(out_ch=512, upsample=False), # 55 -> 53
                # skip-node 5
                dict(out_ch=256*channel_multiplier, upsample=True), # 53 -> 103
                # skip-node 6
                dict(out_ch=256*channel_multiplier, upsample=False), # 103 -> 101
                # skip-node 7
                dict(out_ch=128*channel_multiplier, upsample=True), # 101 -> 199
                # skip-node 8
                dict(out_ch=128*channel_multiplier, upsample=False), # 199 -> 197
                # skip-node 9
            ]
            self.to_rgbs_specs = [
                dict(src=1, tgt=3, upsample=True),
                dict(src=3, tgt=5, upsample=True),
                dict(src=5, tgt=7, upsample=True),
                dict(src=7, tgt=9, upsample=True),
                dict(src=9, tgt=10, upsample=True),
            ]
        elif g_output_res == 389 and config.train_params.ts_input_size == 11:
            self.convs_specs = [
                dict(out_ch=512, upsample=True), # 11 -> 19
                # skip-node 0
                dict(out_ch=512, upsample=False), # 19 -> 17
                # skip-node 1
                dict(out_ch=512, upsample=True), # 17 -> 31
                # skip-node 2
                dict(out_ch=512, upsample=False), # 31 -> 29
                # skip-node 3
                dict(out_ch=512, upsample=True), # 29 -> 55
                # skip-node 4
                dict(out_ch=512, upsample=False), # 55 -> 53
                # skip-node 5
                dict(out_ch=256*channel_multiplier, upsample=True), # 53 -> 103
                # skip-node 6
                dict(out_ch=256*channel_multiplier, upsample=False), # 103 -> 101
                # skip-node 7
                dict(out_ch=128*channel_multiplier, upsample=True), # 101 -> 199
                # skip-node 8
                dict(out_ch=128*channel_multiplier, upsample=False), # 199 -> 197
                # skip-node 9
                dict(out_ch=64*channel_multiplier, upsample=True), # 197 -> 391
                # skip-node 10
                dict(out_ch=64*channel_multiplier, upsample=False), # 391 -> 389
                # skip-node 11
            ]
            self.to_rgbs_specs = [
                dict(src=1, tgt=3, upsample=True),
                dict(src=3, tgt=5, upsample=True),
                dict(src=5, tgt=7, upsample=True),
                dict(src=7, tgt=9, upsample=True),
                dict(src=9, tgt=11, upsample=True),
                dict(src=11, tgt=12, upsample=True),
            ]
        elif g_output_res == 773 and config.train_params.ts_input_size == 11:
            self.convs_specs = [
                dict(out_ch=512, upsample=True), # 11 -> 19
                # skip-node 0
                dict(out_ch=512, upsample=False), # 19 -> 17
                # skip-node 1
                dict(out_ch=512, upsample=True), # 17 -> 31
                # skip-node 2
                dict(out_ch=512, upsample=False), # 31 -> 29
                # skip-node 3
                dict(out_ch=512, upsample=True), # 29 -> 55
                # skip-node 4
                dict(out_ch=512, upsample=False), # 55 -> 53
                # skip-node 5
                dict(out_ch=256*channel_multiplier, upsample=True), # 53 -> 103
                # skip-node 6
                dict(out_ch=256*channel_multiplier, upsample=False), # 103 -> 101
                # skip-node 7
                dict(out_ch=128*channel_multiplier, upsample=True), # 101 -> 199
                # skip-node 8
                dict(out_ch=128*channel_multiplier, upsample=False), # 199 -> 197
                # skip-node 9
                dict(out_ch=64*channel_multiplier, upsample=True), # 197 -> 391
                # skip-node 10
                dict(out_ch=64*channel_multiplier, upsample=False), # 391 -> 389
                # skip-node 11
                dict(out_ch=32*channel_multiplier, upsample=True), # 389 -> 775
                # skip-node 12
                dict(out_ch=32*channel_multiplier, upsample=False), # 775 -> 773
                # skip-node 13
            ]
            self.to_rgbs_specs = [
                dict(src=1, tgt=3, upsample=True),
                dict(src=3, tgt=5, upsample=True),
                dict(src=5, tgt=7, upsample=True),
                dict(src=7, tgt=9, upsample=True),
                dict(src=9, tgt=11, upsample=True),
                dict(src=11, tgt=13, upsample=True),
                dict(src=13, tgt=14, upsample=True),
            ]
        elif g_output_res == 1541 and config.train_params.ts_input_size == 11:
            self.convs_specs = [
                dict(out_ch=512, upsample=True), # 11 -> 19
                # skip-node 0
                dict(out_ch=512, upsample=False), # 19 -> 17
                # skip-node 1
                dict(out_ch=512, upsample=True), # 17 -> 31
                # skip-node 2
                dict(out_ch=512, upsample=False), # 31 -> 29
                # skip-node 3
                dict(out_ch=512, upsample=True), # 29 -> 55
                # skip-node 4
                dict(out_ch=512, upsample=False), # 55 -> 53
                # skip-node 5
                dict(out_ch=256*channel_multiplier, upsample=True), # 53 -> 103
                # skip-node 6
                dict(out_ch=256*channel_multiplier, upsample=False), # 103 -> 101
                # skip-node 7
                dict(out_ch=128*channel_multiplier, upsample=True), # 101 -> 199
                # skip-node 8
                dict(out_ch=128*channel_multiplier, upsample=False), # 199 -> 197
                # skip-node 9
                dict(out_ch=64*channel_multiplier, upsample=True), # 197 -> 391
                # skip-node 10
                dict(out_ch=64*channel_multiplier, upsample=False), # 391 -> 389
                # skip-node 11
                dict(out_ch=32*channel_multiplier, upsample=True), # 389 -> 775
                # skip-node 12
                dict(out_ch=32*channel_multiplier, upsample=False), # 775 -> 773
                # skip-node 13
                dict(out_ch=16*channel_multiplier, upsample=True), # 773 -> 1543
                # skip-node 14
                dict(out_ch=16*channel_multiplier, upsample=False), # 1543 -> 1541
                # skip-node 15
            ]
            self.to_rgbs_specs = [
                dict(src=1, tgt=3, upsample=True),
                dict(src=3, tgt=5, upsample=True),
                dict(src=5, tgt=7, upsample=True),
                dict(src=7, tgt=9, upsample=True),
                dict(src=9, tgt=11, upsample=True),
                dict(src=11, tgt=13, upsample=True),
                dict(src=13, tgt=15, upsample=True),
                dict(src=15, tgt=16, upsample=True),
            ]
        elif g_output_res == 128 and config.train_params.ts_input_size == 4: # Basline
            self.convs_specs = [
                dict(out_ch=512, upsample=True), # 4 -> 8
                # skip-node 0
                dict(out_ch=512, upsample=False),
                # skip-node 1
                dict(out_ch=512, upsample=True), # 8 -> 16
                # skip-node 2
                dict(out_ch=512, upsample=False),
                # skip-node 3
                dict(out_ch=512, upsample=True), # 16 -> 32
                # skip-node 4
                dict(out_ch=512, upsample=False),
                # skip-node 5
                dict(out_ch=512, upsample=True), # 32 -> 64
                # skip-node 6
                dict(out_ch=512, upsample=False),
                # skip-node 7
                dict(out_ch=256 * channel_multiplier, upsample=True), # 64 -> 128
                # skip-node 8
                dict(out_ch=256 * channel_multiplier, upsample=False),
                # skip-node 9
            ]
            self.to_rgbs_specs = [
                dict(src=1, tgt=3,  upsample=True),
                dict(src=3, tgt=5,  upsample=True),
                dict(src=5, tgt=7,  upsample=True),
                dict(src=7, tgt=9,  upsample=True),
                dict(src=9, tgt=10,  upsample=True),
            ]
        elif g_output_res == 64 and config.train_params.ts_input_size == 4: # Basline
            self.convs_specs = [
                dict(out_ch=512, upsample=True), # 4 -> 8
                # skip-node 0
                dict(out_ch=512, upsample=False),
                # skip-node 1
                dict(out_ch=512, upsample=True), # 8 -> 16
                # skip-node 2
                dict(out_ch=512, upsample=False),
                # skip-node 3
                dict(out_ch=512, upsample=True), # 16 -> 32
                # skip-node 4
                dict(out_ch=512, upsample=False),
                # skip-node 5
                dict(out_ch=512, upsample=True), # 32 -> 64
                # skip-node 6
                dict(out_ch=512, upsample=False),
                # skip-node 7
            ]
            self.to_rgbs_specs = [
                dict(src=1, tgt=3,  upsample=True),
                dict(src=3, tgt=5,  upsample=True),
                dict(src=5, tgt=7,  upsample=True),
                dict(src=7, tgt=8,  upsample=True),
            ]
        else:
            raise NotImplementedError(" Not yet designed arch for G input size {} and output res {}".format(
                config.train_params.ts_input_size, g_output_res))

        self.const_z = ConstantInput(self.local_latent_dim)

        self.num_layers = len(self.convs_specs)
        self.n_latent = self.num_layers + 1 # The last latent is for the additional to_rgb skip only

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        in_ch = self.local_latent_dim

        for i,conv_spec in enumerate(self.convs_specs):

            self.convs.append(
                StyledConv(
                    in_ch, 
                    conv_spec["out_ch"], 
                    3, 
                    self.global_latent_dim, 
                    upsample=conv_spec["upsample"],
                    blur_kernel=blur_kernel,
                    no_zero_pad=config.train_params.ts_no_zero_pad,
                    config=config,
                    side="ts")
            )
            in_ch = conv_spec["out_ch"]

        for to_rgb_spec in self.to_rgbs_specs:
            src_conv_spec = self.convs_specs[to_rgb_spec["src"]]
            in_ch = src_conv_spec["out_ch"]
            self.to_rgbs.append(
                ToRGB(
                    in_ch,
                    self.global_latent_dim, 
                    upsample=to_rgb_spec["upsample"],
                    no_zero_pad=config.train_params.ts_no_zero_pad,
                    blur_kernel=blur_kernel,
                    config=config,
                    side="ts",
                )
            )

    def calc_in_spatial_size(self, out_spatial_size, return_list=False):
        # print(" [*] Calculating input spatial size:")
        in_spatial_sizes = []
        for i,conv in enumerate(self.convs[::-1]):
            if conv.conv.upsample:
                conv_type = "upsample"
            # elif conv.conv.downsample:
            #     conv_type = "downsample"
            else:
                conv_type = "normal"
            in_spatial_size = conv.calc_in_spatial_size(out_spatial_size)
            # print("\t {}-th layer ({}), output {} => input {}".format(len(self.convs)-i, conv_type, out_spatial_size, in_spatial_size))
            out_spatial_size = in_spatial_size
            in_spatial_sizes.append(in_spatial_size)
        if return_list:
            return in_spatial_sizes[::-1] # retain the z->ss->ts->img order
        else:
            return in_spatial_sizes[-1]

    def calc_out_spatial_size(self, in_spatial_size, return_list=False):
        # print(" [*] Calculating output spatial size:")
        out_spatial_sizes = []
        for i,conv in enumerate(self.convs):
            if conv.conv.upsample:
                conv_type = "upsample"
            # elif conv.conv.downsample:
            #     conv_type = "downsample"
            else:
                conv_type = "normal"
            out_spatial_size = conv.calc_out_spatial_size(in_spatial_size)
            # print("\t {}-th layer ({}), input {} => output {}".format(i, conv_type, in_spatial_size, out_spatial_size))
            in_spatial_size = out_spatial_size
            out_spatial_sizes.append(out_spatial_size)
        
        if return_list:
            return out_spatial_sizes
        else:
            return out_spatial_sizes[-1]

    def calibrate_spatial_shape(self, feature, direction, padding_mode="replicate", verbose=False, pin_loc=None):
        # We only calibrates for the convs in the main stream, 
        # the to_rgb stream can directly use the center-cropped region.
        ret_latents = []
        ret_pin_locs = []
        if direction == "forward":
            ops = self.convs
        elif direction == "backward":
            ops = self.convs[::-1]
        else:
            raise NotImplementedError()
        for i,conv in enumerate(ops):
            feature, pin_loc = \
                conv.calibrate_spatial_shape(feature, direction, padding_mode=padding_mode, verbose=verbose, pin_loc=pin_loc)
            ret_latents.append(feature)
            ret_pin_locs.append(pin_loc)
        return ret_latents, ret_pin_locs

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.global_latent_dim, device=self.const_z.input.device
        )
        latent = self.mapping(latent_in).mean(0, keepdim=True)
        return latent

    def get_style(self, input):
        return self.mapping(input)

    def forward(
        self,
        *args,

        # Possible inputs
        global_latent=None,
        structure_latent=None,
        styles=None,
        wplus_styles=None,
        noises=None,
        style_fusion_map=None,
        test_ids=None,

        # Additional outputs
        return_latents=False,
        return_path_length=False,

        # Control signals
        inject_index=None,
        calc_flops=False,

        **kwargs):

        assert len(args) == 0, "Do not allow non-keyword argument in generator!"
        flops = 0

        """
        Noise inputs (StyleGAN paper section 3.2)
        """
        if noises is None:
            noises = [None] * self.num_layers

        # Style is already given, we assume all mixing stuffs (if desired) are already handled
        if (styles is None) and (wplus_styles is None):

            """
            Mapping
            """
            if global_latent.ndim == 3: # style mixing multiple global latent
                styles = [self.mapping(global_latent[:, i]) for i in range(global_latent.shape[1])]
                if calc_flops: # disregard style mixing
                    _, flops_mapping = sequential_get_flops(self.mapping, global_latent)
            else:
                if calc_flops:
                    styles, flops_mapping = sequential_get_flops(self.mapping, global_latent)
                    styles = [styles]
                    flops += flops_mapping
                else:
                    styles = [self.mapping(global_latent)]

            """
            Truncation
            """
            # # No cheating plz OuO
            # The latent space is what you are modeling. You should never trade recall by eliminating a part of it.
            # For instance, like in the image classification task, can you refuse to classify testing samples with low confidence?

            """
            Style Mixing
            """
            if global_latent.ndim == 3: # style mixing multiple global latent
                if inject_index is None:
                    if self.training:
                        inject_index = random.randint(1, self.n_latent - 1)
                    else:
                        inject_index = self.n_latent
                if inject_index == self.n_latent:
                    styles = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                else:
                    styles = torch.cat([
                        styles[0].unsqueeze(1).repeat(1, inject_index, 1),
                        styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1),
                    ], 1)
            else:
                inject_index = self.n_latent
                if styles[0].ndim < 3:
                    styles = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                else:
                    styles = styles[0]
        

        # Spatial style fusion
        if style_fusion_map is not None:
            if wplus_styles is None:
                assert calc_flops == False
                convs_fused_styles = [
                    create_fusion_styles(fm, styles, self.config) for fm in style_fusion_map]
                torgbs_fused_styles = []
                for to_rgbs_spec in self.to_rgbs_specs:
                    cur_fusion_map = style_fusion_map[to_rgbs_spec["src"]]
                    fused_style = create_fusion_styles(cur_fusion_map, styles, self.config)
                    torgbs_fused_styles.append(fused_style)
            else:
                assert calc_flops == False
                convs_fused_styles = []
                for i in range(len(style_fusion_map)):
                    cur_wplus_style_centers = [
                        s[:, i] for s in wplus_styles]
                    cur_fusion_map = create_fusion_styles(
                        style_fusion_map[i], cur_wplus_style_centers, self.config)
                    convs_fused_styles.append(cur_fusion_map)

                torgbs_fused_styles = []
                for to_rgbs_spec in self.to_rgbs_specs:
                    src_idx = to_rgbs_spec["src"]
                    tgt_idx = to_rgbs_spec["tgt"]
                    cur_fusion_map = style_fusion_map[src_idx]
                    cur_wplus_style_centers = [
                        s[:, tgt_idx] for s in wplus_styles]
                    fused_style = create_fusion_styles(
                        cur_fusion_map, cur_wplus_style_centers, self.config)
                    torgbs_fused_styles.append(fused_style)

        """
        Synthesis image from styles
        """
        cur_to_rgb_idx = 0
        skip = None
        h = structure_latent
        upsample_between_skips = True # The first ToRGB does not take skip input, make it always valid
        for i,(conv,noise) in enumerate(zip(self.convs, noises)):

            # Deal with different types of input
            if style_fusion_map is not None: # Supposts spatially-shaped style at testing
                cur_style = convs_fused_styles[i]
            elif wplus_styles is not None:
                cur_style = wplus_styles[:, i]
            elif isinstance(styles, list):
                cur_style = styles[i]
            else: # Native
                cur_style = styles[:, i]

            h, cur_flops = conv(h, cur_style, noise=noise, test_ids=test_ids, calc_flops=calc_flops)
            flops += cur_flops

            if conv.upsample: 
                # Safety check, since my implementation trusts all the feature shapes are correctly justified, 
                #  then automatically performs shape callibrations on shape mismatches.
                upsample_between_skips = True


            to_rgb_spec = self.to_rgbs_specs[cur_to_rgb_idx]
            skip_src = to_rgb_spec["src"]
            skip_tgt = to_rgb_spec["tgt"]
            if i == skip_src:
                assert upsample_between_skips, \
                    "At least one upsampling conv between two consecutive skip" + \
                    " layers (ToRGB) is required. Failed to meet the requirement" + \
                    " before ToRGB ({} => {})".format(skip_src, skip_tgt)
                to_rgb_op = self.to_rgbs[cur_to_rgb_idx]
                
                # Deal with different types of input
                if style_fusion_map is not None: # Supposts spatially-shaped style at testing
                    cur_style = torgbs_fused_styles[cur_to_rgb_idx]
                elif wplus_styles is not None:
                    cur_style = wplus_styles[:, skip_tgt]
                elif isinstance(styles, list):
                    cur_style = styles[skip_tgt]
                else: # Native
                    cur_style = styles[:, skip_tgt]

                skip, cur_flops = to_rgb_op(h, cur_style, skip=skip, calc_flops=calc_flops)
                flops += cur_flops

                cur_to_rgb_idx += 1
                upsample_between_skips = False

        image = skip


        """
        Output
        """
        output = DictTensor(gen=image)

        if return_latents:
            output["latents"] = styles
        if return_path_length:
            output["path_lengths"] = calc_path_lengths(image, [styles])
        if calc_flops:
            output["flops_ts"] = flops
        return output


class InfinityGanGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if hasattr(config.train_params, "styleGAN2_baseline") and config.train_params.styleGAN2_baseline:
            if hasattr(config.train_params, "force_use_ss") and config.train_params.force_use_ss:
                self.structure_synthesizer = StructureSynthesizer(config)
            else:
                self.structure_synthesizer = None
        elif config.train_params.use_ss:
            self.structure_synthesizer = StructureSynthesizer(config)
        else:
            self.structure_synthesizer = None
        self.texture_synthesizer = TextureSynthesizer(config)

        if hasattr(self.config.train_params, "diversity_z_w") and self.config.train_params.diversity_z_w != 0:
            self.use_div_z = True
        else:
            self.use_div_z = False

        if hasattr(self.config.train_params, "diversity_angular") and self.config.train_params.diversity_angular:
            self.use_angular_div = True
        else:
            self.use_angular_div = False

    def _diversity_dist(self, values):
        if self.use_angular_div:
            if values.shape[0] % 2 == 0:
                x_dist = angular_similarity(values[0::2], values[1::2]).mean()
            else: # Deal with cases that batch size is odd due to nn.Dataparallel
                x_dist = angular_similarity(values[0:-1:2], values[1::2]).mean()
            return x_dist # already in [0, 1]
        else:
            if values.shape[0] % 2 == 0:
                x_dist = (values[0::2] - values[1::2]).abs().mean()
            else: # Deal with cases that batch size is odd due to nn.Dataparallel
                x_dist = (values[0:-1:2] - values[1::2]).abs().mean()
            return x_dist

    def calc_in_spatial_size(self, out_spatial_size, include_ss=False, return_list=False):
        assert include_ss == False, "Never used. GNN padding is always explicitly handled outside."
        in_spatial_sizes_ts = self.texture_synthesizer.calc_in_spatial_size(
            out_spatial_size, return_list=return_list)

        if include_ss:
            out_spatial_size = in_spatial_sizes_ts[0] if return_list else in_spatial_sizes_ts
            in_spatial_sizes_ss = self.structure_synthesizer.calc_in_spatial_size(
                out_spatial_size, return_list=return_list)
            if return_list:
                return [int(np.ceil(v)) for v in in_spatial_sizes_ss + in_spatial_sizes_ts]
            else:
                return int(np.ceil(in_spatial_sizes_ss))
        else:
            if return_list:
                return [int(np.ceil(v)) for v in in_spatial_sizes_ts]
            else:
                return int(np.ceil(in_spatial_sizes_ts))
        
    def calc_out_spatial_size(self, in_spatial_size, include_ss=False, return_list=False):
        # assert include_ss == False, "Never used. GNN padding is always explicitly handled outside."
        if include_ss: # GNN padding are usually already accounted elsewhere
            out_spatial_sizes_ss = self.structure_synthesizer.calc_out_spatial_size(
                in_spatial_size, return_list=return_list)
            ts_in_spatial_size = out_spatial_sizes_ss[-1] if return_list else out_spatial_sizes_ss
            out_spatial_sizes_ts = self.texture_synthesizer.calc_out_spatial_size(
                ts_in_spatial_size, return_list=return_list)
        else:
            out_spatial_sizes_ss = []
            out_spatial_sizes_ts = self.texture_synthesizer.calc_out_spatial_size(
                in_spatial_size, return_list=return_list)

        if return_list:
            return [int(np.ceil(v)) for v in out_spatial_sizes_ss + out_spatial_sizes_ts]
        else:
            return int(np.ceil(out_spatial_sizes_ts))

    def calibrate_spatial_shape(self, feature, direction, padding_mode="replicate", verbose=False, pin_loc=None):
        if direction == "forward":
            feature_list_ss, pin_loc_list_ss = \
                self.structure_synthesizer.calibrate_spatial_shape(
                    feature,
                    direction, 
                    padding_mode=padding_mode,
                    verbose=verbose,
                    pin_loc=pin_loc)
            feature_list_ts, pin_loc_list_ts = \
                self.texture_synthesizer.calibrate_spatial_shape(
                    feature_list_ss[-1], # order is `SS input => SS output`
                    direction, 
                    padding_mode=padding_mode,
                    verbose=verbose,
                    pin_loc=pin_loc_list_ss[-1] if pin_loc is not None else None)
            return feature_list_ss, feature_list_ts, pin_loc_list_ss, pin_loc_list_ts
        elif direction == "backward":
            feature_list_ts, pin_loc_list_ts = \
                self.texture_synthesizer.calibrate_spatial_shape(
                    feature, 
                    direction, 
                    padding_mode=padding_mode,
                    verbose=verbose,
                    pin_loc=pin_loc)
            feature_list_ss, pin_loc_list_ss = \
                self.structure_synthesizer.calibrate_spatial_shape(
                    feature_list_ts[-1], # order is `TS output => TS input`
                    direction, 
                    padding_mode=padding_mode,
                    verbose=verbose,
                    pin_loc=pin_loc_list_ts[-1] if pin_loc is not None else None)
            return feature_list_ss[::-1], feature_list_ts[::-1], pin_loc_list_ss[::-1], pin_loc_list_ts[::-1]
        else:
            raise NotImplementedError()

    def get_style(self, global_latent):
        return self.texture_synthesizer.get_style(global_latent)

    def forward(
        self,
        *args,

        # Possible inputs
        global_latent=None,
        local_latent=None,
        structure_latent=None,
        styles=None,
        wplus_styles=None,
        noises=None,
        ss_noises=None,
        override_coords=None,
        test_ids=None,

        # Additional outputs
        return_latents=False,
        return_path_length=False,
        early_return_structure_latent=False,

        # Control signals
        inject_index=None,
        disable_dual_latents=False,
        is_fid_eval=False,
        calc_flops=False,
        style_fusion_map_ss=None,
        style_fusion_map_ts=None,
        call_internal_method=None,
        internal_method_kwargs={},
        **kwargs):

        if call_internal_method is not None:
            return getattr(self, call_internal_method)(**internal_method_kwargs)

        assert ("coords" not in kwargs) and ("testing_coords" not in kwargs), \
            "Depricated argument, should use `override_coords` instead!"

        # Get device
        if global_latent is not None:
            if isinstance(global_latent, list):
                device = global_latent[0].device
            else:
                device = global_latent.device
        else:
            if isinstance(styles, list):
                device = styles[0].device
            else:
                device = styles.device

        """
        Setup dual global_latent for diversity_z_loss
        """
        if self.use_div_z:
            if disable_dual_latents:
                pass
            else:
                assert self.training
                global_latent = setup_paired_inputs(global_latent)

        """
        Structure Synthesizer
        """
        if (local_latent is not None) and (self.structure_synthesizer is not None):
            structure_latent, coords, ac_coords, flops_ss = self.structure_synthesizer(
                global_latent=global_latent if isinstance(global_latent, list) else global_latent[:, 0],
                local_latent=local_latent, 
                override_coords=override_coords,
                noises=ss_noises,
                test_ids=test_ids,
                disable_dual_latents=disable_dual_latents,
                is_fid_eval=is_fid_eval,
                style_fusion_map=style_fusion_map_ss,
                calc_flops=calc_flops)
        else:
            ac_coords = None

        # Cases without structure synthesizer, like baseline StyleGAN2
        if self.structure_synthesizer is None:
            if local_latent is None:
                assert structure_latent is not None # "Directly assigned structure_latent"
            else:
                structure_latent = local_latent
            coords = None
            flops_ss = 0
        else:
            #assert override_coords is not None
            coords = override_coords
            flops_ss = 0


        if early_return_structure_latent:
            return DictTensor(
                structure_latent=structure_latent, 
                coords=coords)


        """
        Texture Synthesizer
        """
        output = self.texture_synthesizer(
            global_latent=global_latent,
            structure_latent=structure_latent,
            styles=styles,
            wplus_styles=wplus_styles,
            noises=noises,
            return_latents=return_latents,
            return_path_length=return_path_length,
            inject_index=inject_index,
            test_ids=test_ids,
            style_fusion_map=style_fusion_map_ts,
            calc_flops=calc_flops)

        """
        Mode-seeking diversity loss
        """
        if self.use_div_z and (not (local_latent is None)) and self.training:
            output["diversity_z_loss"] = \
                self.structure_synthesizer.diversity_z_loss(
                    global_latent[:, 0], local_latent, structure_latent, eps=1e-5)
        else:
            output["diversity_z_loss"] = torch.tensor(0).float().to(device)

        """
        Manage Outputs
        """
        output["structure_latent"] = structure_latent
        if ac_coords is not None:
            output["ac_coords"] = ac_coords

        if calc_flops:
            assert isinstance(flops_ss, int) or isinstance(flops_ss, np.int64), \
                "Got SS flops result with type {}, should be int!".format(type(flops_ss))
            assert isinstance(output["flops_ts"], int) or isinstance(output["flops_ts"], np.int64), \
                "Got TS flops result with type {}, should be int!".format(type(output["flops_ts"]))
            output["flops_ss"] = flops_ss
            output["flops_ts"] = output["flops_ts"]
            output["flops_all"] = output["flops_ss"] + output["flops_ts"]

        return output

