import math
import torch
from torch import nn

from models.ops import *
from dataset import DictTensor


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, 
                 blur_kernel=[1, 3, 3, 1], downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, kernel_size)
        self.conv2 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample)

        self.skip = ConvLayer(
            in_channel, out_channel, kernel_size=1, downsample=downsample, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class StyleGan2Discriminator(nn.Module):
    def __init__(self, config, no_adds_on=False):
        super().__init__()
        self.config = config
        self.size = size = config.train_params.patch_size
        channel_multiplier = config.train_params.channel_multiplier
        blur_kernel = [1, 3, 3, 1]

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
            2048: 8 * channel_multiplier,
        }
        linear_ch = 512

        if hasattr(self.config.train_params, "d_extra_multiplier"):
            linear_ch = round(linear_ch * self.config.train_params.d_extra_multiplier)
            for k in channels:
                channels[k] = round(channels[k] * self.config.train_params.d_extra_multiplier)

        log_size = int(round(math.log(size, 2)))
        convs = [ConvLayer(3, channels[2 ** log_size], kernel_size=1)]

        in_channel = channels[2 ** log_size]

        cur_out_spatial_size = size
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, 3, blur_kernel))
            in_channel = out_channel
            cur_out_spatial_size //= 2
        self.last_feat_ch = in_channel

        if no_adds_on:
            self.use_coord_ac = False
            self.use_coord_pd = False
        else:
            self.use_coord_ac = hasattr(self.config.train_params, "coord_use_ac") and self.config.train_params.coord_use_ac
            self.use_coord_pd = hasattr(self.config.train_params, "coord_use_pd") and self.config.train_params.coord_use_pd

        if self.use_coord_pd:
            self.convs_head= nn.Sequential(*convs[:-1])
            self.convs_tail = convs[-1]
        else:
            self.convs = nn.Sequential(*convs)

        # self.stddev_group = 4
        # Will remain 4 if batch is divisable by 4
        self.stddev_group = self._smallest_divisor_larger_than(config.train_params.batch_size, start=4)
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, linear_ch, kernel_size=3)
        self.final_linear = nn.Sequential(
            EqualLinear(linear_ch * cur_out_spatial_size * cur_out_spatial_size, linear_ch, activation='fused_lrelu'),
            EqualLinear(linear_ch, 1),
        )

        if self.use_coord_ac:
            if hasattr(config.train_params, "coord_ac_categorical") and config.train_params.coord_ac_categorical:
                assert config.train_params.coord_ac_vert_only, "experimental setup"
                self.coord_linear = nn.Sequential(
                    EqualLinear(
                        linear_ch * cur_out_spatial_size * cur_out_spatial_size, 
                        linear_ch, 
                        activation='fused_lrelu'),
                    EqualLinear(linear_ch, config.train_params.coord_num_dir * config.train_params.coord_vert_sample_size),
                )
            else:
                self.coord_linear = nn.Sequential(
                    EqualLinear(
                        linear_ch * cur_out_spatial_size * cur_out_spatial_size, 
                        linear_ch, 
                        activation='fused_lrelu'),
                    EqualLinear(linear_ch, config.train_params.coord_num_dir),
                )
        
        if self.use_coord_pd:
            if hasattr(config.train_params, "coord_pd_hori_only") and config.train_params.coord_pd_hori_only:
                self.coord_proj_dim = config.train_params.coord_num_dir - 1
            else:
                self.coord_proj_dim = config.train_params.coord_num_dir
            self.coord_proj = nn.Sequential(
                EqualLinear(
                    self.coord_proj_dim, 
                    linear_ch, 
                    activation='fused_lrelu'),
                EqualLinear(linear_ch, linear_ch),
            )
    

    def _smallest_divisor_larger_than(self, number, start):
        for i in range(start, int(math.sqrt(number))):
            if number % i == 0:
                return i
        return number # No other divisor other than self
            

    def forward(self, input_data, **kwargs):

        if "feature_disc"in kwargs and kwargs["feature_disc"]:
            return self.feat_discriminator(input_data)

        if isinstance(input_data, torch.Tensor):
            img = input_data
        elif "gen" in input_data:
            img = input_data["gen"]
        else:
            modality = self.config.train_params.training_modality
            img = input_data[modality]


        if self.use_coord_pd:
            h = self.convs_head(img)
            last_feat = h
            h = self.convs_tail(h)
        else:
            h = self.convs(img)


        batch, channel, height, width = h.shape
        group = min(batch, self.stddev_group)
        stddev = h.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        h = torch.cat([h, stddev], 1)

        out = self.final_conv(h).view(batch, -1)
        before_linear = out
        out = self.final_linear(out)


        ret = DictTensor(d_patch=out)
        if self.use_coord_ac:
            ret["ac_coords_pred"] = self.coord_linear(before_linear)
        if self.use_coord_pd and self.training:
            label = input_data["ac_coords"][:, -self.coord_proj_dim:] # shape (B, num_dir)
            label_proj = self.coord_proj(label) # shape (B, C)
            feat_proj = last_feat.sum([2, 3]) # shape (B, C)
            proj_pred = (label_proj * feat_proj).sum(1, keepdim=True) # shape (B, 1)
            ret["d_patch"] = ret["d_patch"] + proj_pred * self.config.train_params.coord_pd_w
            
        return ret
