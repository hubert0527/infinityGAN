import math
import numpy as np
from scipy import signal

import torch
from torch import nn
from torch.nn import functional as F

from models.custom_ops import FusedLeakyReLU, fused_leaky_relu, upfirdn2d



class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def get_flops(self, input):
        return np.prod(input.shape[1:])

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2, no_zero_pad=False):
        super().__init__()
        self.no_zero_pad = no_zero_pad

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        if no_zero_pad:
            self.pad = (0, 0)
        else:
            p = kernel.shape[0] - factor
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2
            self.pad = (pad0, pad1)

        if no_zero_pad:
            self.upsample = nn.Upsample(scale_factor=2, mode="linear")

    def forward(self, input):
        if self.no_zero_pad:
            B, C, H, W = input.shape
            out = F.conv_transpose2d(input.view(B*C, 1, H, W), self.kernel.unsqueeze(0).unsqueeze(1), stride=2)
            _, _, nH, nW = out.shape
            out = out.view(B, C, nH, nW)[:, :, 1:-1, 1:-1]
        else:
            out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out


def create_gaussian_kernel(kernel_size, std=1):
    gkern1d = signal.gaussian(kernel_size, std=std).reshape(kernel_size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, padding_mode="zero", prior="gaussian"):
        super().__init__()

        if isinstance(kernel, int):
            if prior.lower() == "gaussian".lower():
                kernel = create_gaussian_kernel(kernel_size=kernel)
            elif prior.lower() == "mean".lower():
                kernel = torch.ones(kernel, kernel, dtype=torch.float32)
            else:
                raise NotImplementedError("Unknown prior {}".format(prior))

        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer('kernel', kernel)

        if padding_mode == "replicate":
            self.zero_pad = (0, 0)
            self.replicate_pad = pad if isinstance(pad, tuple) else (pad, pad, pad, pad)
            self.use_replicate_pad = True
        elif padding_mode == "zero":
            self.zero_pad = pad if isinstance(pad, tuple) else (pad, pad)
            self.replicate_pad = 0
            self.use_replicate_pad = False
        else:
            raise NotImplementedError("Unknown padding_mode {}".format(padding_mode))
        self.upsample_factor = upsample_factor

    def get_flops(self, shape):
        _, C, H, W = shape
        ks = self.kernel.shape[0]
        _, _, h_num_iters, w_num_iters = self.get_output_shape(shape)
        return h_num_iters * w_num_iters * C * (ks**2)

    def get_output_shape(self, shape):
        B, C, H, W = shape
        ks = self.kernel.shape[0]
        if self.use_replicate_pad:
            H += self.replicate_pad[2] + self.replicate_pad[3]
            W += self.replicate_pad[0] + self.replicate_pad[1]
        else:
            H += self.zero_pad[0]
            W += self.zero_pad[1]
        out_h = (H - ks//2*2)
        out_w = (W - ks//2*2)
        return B, C, out_h, out_w

    def forward(self, input):
        if self.use_replicate_pad:
            input = F.pad(input, self.replicate_pad, mode="replicate")
        out = upfirdn2d(input, self.kernel, pad=self.zero_pad)
        return out


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding_type = padding
        self.extra_padding_layer = None
        if type(padding) is str:
            if padding == "reflect":
                # Pytorch reflection padding is broken in some early versions.
                self.extra_padding_layer = nn.ReflectionPad2d(kernel_size//2)
                self.zero_pad_size = 0
            elif padding == "zero":
                self.zero_pad_size = kernel_size//2
            else:
                raise NotImplementedError("Unknown padding type {}".format(padding))
        else:
            self.zero_pad_size = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        if self.extra_padding_layer is not None:
            input = self.extra_padding_layer(input)
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.zero_pad_size)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding_type={self.padding_type})')


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def get_flops(self, input):
        flops = 0
        if self.activation:
            flops += self.bias.shape[0] + self.bias.shape[0] # activation
        flops += np.prod(self.weight.shape) + np.prod(self.weight.shape) # linear weight, includes self.scale
        flops += self.bias.shape[0] + self.bias.shape[0] # linear bias, includes self.lr_mul
        return flops

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 2, 1],
        no_zero_pad=False,
        config=None,
        side=None,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.style_dim = style_dim
        self.upsample = upsample
        self.downsample = downsample
        self.no_zero_pad = no_zero_pad
        self.config = config
        self.side = side

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)

        if upsample:
            factor = 2
            assert kernel_size == 3, \
                "I'm only sure about the implementation of 3x3 kernels, " + \
                "the sizes may (VERY LIKELY) have size handling bugs."
            if len(blur_kernel) % 2 == 1: # Ours
                pad0 = pad1 = len(blur_kernel) // 2
            else: # Original StyleGAN2
                # [Hubert] Logic from: https://github.com/rosinality/stylegan2-pytorch/blob/3dee637b8937bf3830991c066ed8d9cc58afd661/model.py#L191
                # I can't really decode the logic, but seems generally right for even-sized blur_kernel.
                p = (len(blur_kernel) - factor) - (kernel_size - 1)
                pad0 = (p + 1) // 2 + factor - 1
                pad1 = p // 2 + 1
            if no_zero_pad:
                self.dirty_rm_size = (pad0, pad1) # The dirty pixels going to be cut-off from features during runtime
                self.blur = Blur(blur_kernel, pad=(0, 0), upsample_factor=factor)
            else:
                self.dirty_rm_size = (0, 0)
                self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
        elif downsample:
            raise NotImplementedError("Never used.")
        else:
            if no_zero_pad:
                self.padding = 0 # Directly use input and cut off
                self.dirty_rm_size = (kernel_size//2, kernel_size//2)
            else:
                self.padding = kernel_size // 2
                self.dirty_rm_size = (0, 0)

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.demodulate = demodulate
        if style_dim > 0:
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        else:
            self.modulation = None


    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )


    def calc_in_spatial_size(self, out_spatial_size, verbose=False):
        """
        Calculate the "minimum" input spatial size that covers the given `out_spatial_size` after `forward()`.

        [Note] Here has some rounding involved (you see a `//` and `round_to_even` here). 
        As a result, if you `forward()` with the yielded `in_spatial_size`, 
        it is expected that the resulting output feature is equal-or-larger than `out_spatial_size`.
        """
        if self.upsample:
            round_to_even = lambda v: v if v % 2 == 0 else v + 1
            in_spatial_size = round_to_even(out_spatial_size + 1 + self.dirty_rm_size[0] + self.dirty_rm_size[1]) // 2
            if verbose:
                print("\t\t {} = ({} + {} + {}) // 2".format(in_spatial_size, out_spatial_size, self.dirty_rm_size[0], self.dirty_rm_size[1]))
        elif self.downsample:
            assert False, "Never used, not sure if exists any problem here."
            in_spatial_size = (out_spatial_size + self.dirty_rm_size[0] + self.dirty_rm_size[1]) * 2
            if verbose:
                print("\t\t {} = ({} + {} + {}) * 2".format(in_spatial_size, out_spatial_size, self.dirty_rm_size[0], self.dirty_rm_size[1]))
        else:
            in_spatial_size = out_spatial_size + self.dirty_rm_size[0] + self.dirty_rm_size[1]
            if verbose:
                print("\t\t {} = {} + {} + {}".format(in_spatial_size, out_spatial_size, self.dirty_rm_size[0], self.dirty_rm_size[1]))
        return in_spatial_size


    def calc_out_spatial_size(self, in_spatial_size):
        """
        Calculate the output spatial size when the module is `forward()`-ed with a feature with shape `(B, C, in_spatial_size, in_spatial_size)`.
        """
        if self.upsample:
            out_spatial_size = in_spatial_size * 2 - 1 - self.dirty_rm_size[0] - self.dirty_rm_size[1]
        elif self.downsample:
            assert False, "Never used, not sure if exists any problem here."
            out_spatial_size = in_spatial_size // 2 - self.dirty_rm_size[0] - self.dirty_rm_size[1]
        else:
            out_spatial_size = in_spatial_size - self.dirty_rm_size[0] - self.dirty_rm_size[1]
        return out_spatial_size


    def calibrate_spatial_shape(self, feature, direction, padding_mode="replicate", verbose=False, pin_loc=None):
        """
        With a given `direction` ("forward" or "backward"), generate the reversed `feature` 
        that preserves the correct spatial alignment by discounting the up/down-sampling and padding.

        `pin_loc`: tuple(<int>, <int>)
        In certain cases, with a designated pixel in the image space, we need to identify the (one) pixel in each layer that covers the pixel in the image space.

        [Note] 
        This function is not intended to use in formal generation (i.e., `forward()`), it is frequently used in 
        "spatial style fusion", "outpainting with inverted latent variables", and "interactive generation."
        It is used to reverse-engineer the geometrical patterns designated in the pixel space.
        More specifically, for "spatial style fusion", there exists a (blurry-)boundary between styles, 
        this function reverse and maintains the boundary of the styles at the approximately (not exact due to rounding) same position.

        [Note 2]
        And yes, the `direction="forward"` is never used LoL
        """
        _, _, h, w = feature.shape
        prev_pin_loc = pin_loc # For debugging

        if direction == "forward":
            if self.upsample:
                new_h = h * 2 - 1 # align_corners=True already discounted the extra pixel created by conv_transpose
                new_w = w * 2 - 1
                feature = F.interpolate(
                    feature, size=[new_h, new_w], mode="bilinear", align_corners=True)
                feature = feature[:, :, 1:-1, 1:-1] # discounting blur

                if pin_loc is not None:
                    old_center = [h//2, w//2]
                    new_center = [new_h//2, new_w//2]
                    pin_loc_to_old_center = [
                        pin_loc[0]-old_center[0], 
                        pin_loc[1]-old_center[1]]
                    pin_loc_to_new_center = [
                        pin_loc_to_old_center[0] * 2,
                        pin_loc_to_old_center[1] * 2]
                    pin_loc = [
                        pin_loc_to_new_center[0] + new_center[0],
                        pin_loc_to_new_center[1] + new_center[1]]
            elif self.downsample:
                raise NotImplementedError("Never used.")
            else:
                if (self.padding == 0):
                    assert self.dirty_rm_size[0] != 0 and self.dirty_rm_size[1] != 0
                    feature = feature[:, :, self.dirty_rm_size[0]:-self.dirty_rm_size[0], self.dirty_rm_size[1]:-self.dirty_rm_size[1]]
                if pin_loc is not None:
                    pin_loc = [
                        pin_loc[0] - self.dirty_rm_size[0],
                        pin_loc[1] - self.dirty_rm_size[1],
                    ]
            if verbose:
                if pin_loc is None:
                    print(" [*] Calibration: ({}, {}) => ({}, {})".format(
                        h, w, feature.shape[2], feature.shape[3]))
                else:
                    print(" [*] Calibration: ({}, {}) => ({}, {}) ; Pin: ({}, {}) => ({}, {})".format(
                        h, w, feature.shape[2], feature.shape[3],
                        prev_pin_loc[0], prev_pin_loc[1], pin_loc[0], pin_loc[1]))
        elif direction == "backward":
            recovered_input_size = (self.calc_in_spatial_size(h), self.calc_in_spatial_size(w))
            if self.upsample:
                # [Hubert] Recall the real `forward` procedure is:
                # 
                #          (Input)
                # |   O   O   O   O   O   |
                # => [a]: conv_transpose view
                # | X O X O X O X O X O X |
                # => [b]: after conv_tranpose
                # | D O O O O O O O O O D |
                # => [c]: Then blur
                # | D B O O O O O O O B D | 
                # => [d]: Remove dirty pixels
                #     | O O O O O O O |
                #         (Output)
                # 
                # Let's reverse this QuQ

                # [d] Add dirty padding back
                if (self.dirty_rm_size != (0,0)):
                    feature = F.pad(
                        feature, 
                        (self.dirty_rm_size[1], self.dirty_rm_size[1], self.dirty_rm_size[0], self.dirty_rm_size[0]), # (L, R, T, B)
                        mode=padding_mode)
                # [c][b] No changes in spatial size, no need to deal with. (Note: The "D" pixel in [b][c] steps are ignored)
                # [a] Interpolate to reverse the conv_transpose
                feature = F.interpolate(
                    feature, 
                    size=recovered_input_size, 
                    mode="bilinear",
                    align_corners=True)

                if pin_loc is not None:
                    # [d]
                    pin_loc = [
                        pin_loc[0] + self.dirty_rm_size[0],
                        pin_loc[1] + self.dirty_rm_size[1]]
                    # [c][b] skip
                    # [a]
                    old_center = [ # old center at [b] step
                        h+self.dirty_rm_size[0], w+self.dirty_rm_size[1]]
                    new_center = [ # new center at [a] step
                        old_center[0]//2, old_center[1]//2]
                    pin_loc_to_old_center = [
                        pin_loc[0]-old_center[0], 
                        pin_loc[1]-old_center[1]]
                    pin_loc_to_new_center = [
                        pin_loc_to_old_center[0] // 2,
                        pin_loc_to_old_center[1] // 2]
                    pin_loc = [
                        pin_loc_to_new_center[0] + new_center[0],
                        pin_loc_to_new_center[1] + new_center[1]]
            elif self.downsample:
                raise NotImplementedError("Never used.")
            else:
                # Add padding back due to the lost in Conv2d
                if (self.padding == 0):
                    feature = F.pad(
                        feature, 
                        (self.dirty_rm_size[1], self.dirty_rm_size[1], self.dirty_rm_size[0], self.dirty_rm_size[0]), # (L, R, T, B)
                        mode=padding_mode)
                if pin_loc is not None:
                    pin_loc = [
                        pin_loc[0] + self.dirty_rm_size[0],
                        pin_loc[1] + self.dirty_rm_size[1],
                    ]
            if verbose:
                if pin_loc is None:
                    print(" [*] Calibration: ({}, {}) => ({}, {})".format(h, w, feature.shape[2], feature.shape[3]))
                else:
                    print(" [*] Calibration: ({}, {}) => ({}, {}) ; Pin: ({}, {}) => ({}, {})".format(
                        h, w, feature.shape[2], feature.shape[3],
                        prev_pin_loc[0], prev_pin_loc[1], pin_loc[0], pin_loc[1]))
        else:
            raise NotImplementedError("Unknown direction {} (valid: 'forward' or 'backward')".format(direction))

        return feature, pin_loc

    def _auto_shape_align(self, source=None, target=None):
        assert source.shape[2] >= target.shape[2], \
            "Got shape, source: {}, target: {}".format(source.shape, target.shape)
        assert source.shape[3] >= target.shape[3], \
            "Got shape, source: {}, target: {}".format(source.shape, target.shape)
        assert (source.shape[2] - target.shape[2]) % 2 == 0
        assert (source.shape[3] - target.shape[3]) % 2 == 0
        pad_h = (source.shape[2] - target.shape[2]) // 2
        pad_w = (source.shape[3] - target.shape[3]) // 2
        return source[:, :, pad_h:pad_h+target.shape[2], pad_w:pad_w+target.shape[3]]

    def get_flops(self, input, style):
        flops = 0 # per sample, so do not consider batch dim
        _, in_ch, H, W = input.shape
        _, style_ch = style.shape
        out_ch = self.out_channel
        w_num_params = np.prod(self.weight.shape)

        # style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        if self.modulation is not None:
            flops += self.modulation.get_flops(None)
        # # (1, ) * (1, out_ch, in_ch, k, k) * (B, 1, in_ch, 1, 1)
        # # => (B, out_ch, in_ch, k, k)
        # weight = self.scale * self.weight * style
        flops += w_num_params + w_num_params * style_ch

        if self.demodulate:
            # demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            flops += w_num_params
            # weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
            flops += w_num_params * in_ch

        # weight = weight.view(
        #     batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        # )

        if self.upsample:
            # input = input.view(1, batch * in_channel, height, width)
            # weight = weight.view(
            #     batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            # )
            # weight = weight.transpose(1, 2).reshape(
            #     batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            # )
            # out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            h_num_iters = (H * 2 + 1)
            w_num_iters = (W * 2 + 1)
            flops += w_num_params * h_num_iters * w_num_iters
            # if self.no_zero_pad:
            #     out = out[:, :, 1:-1, 1:-1] # Clipping head and tail, which involves zero padding
            # _, _, height, width = out.shape
            # out = out.view(batch, self.out_channel, height, width)
            # out = self.blur(out)
            out_h = H * 2 + 1
            out_w = W * 2 + 1
            if self.no_zero_pad:
                out_h -= 2
                out_w -= 2
            flops += self.blur.get_flops(shape=(-1, out_ch, out_h, out_w))

        elif self.downsample:
            # input = self.blur(input)
            flops += self.blur.get_flops(shape=(-1, in_ch, H, W))
            # _, _, height, width = input.shape
            # input = input.view(1, batch * in_channel, height, width)
            # out = F.conv2d(input, weight, padding=self.padding, stride=2, groups=batch)
            _, _, cur_h, cur_w = self.blur.get_output_shape(shape=(-1, in_ch, H, W))
            cur_h += self.padding * 2
            cur_w += self.padding * 2
            h_num_iters = (cur_h - self.kernel_size//2*2) // 2 + 1
            w_num_iters = (cur_w - self.kernel_size//2*2) // 2 + 1
            flops += w_num_params * (h_num_iters*w_num_iters)
            # _, _, height, width = out.shape
            # out = out.view(batch, self.out_channel, height, width)
        else:
            # input = input.view(1, batch * in_channel, height, width)
            # out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            H += self.padding * 2
            W += self.padding * 2
            h_num_iters = (H - self.kernel_size//2*2)
            w_num_iters = (W - self.kernel_size//2*2)
            flops += w_num_params * (h_num_iters*w_num_iters)
            # _, _, height, width = out.shape
            # out = out.view(batch, self.out_channel, height, width)

        # print(input.shape, self.weight.shape, w_num_params * (h_num_iters*w_num_iters))
        return flops
        

    def forward(self, input, style, coords=None, calc_flops=False):
        batch, in_channel, height, width = input.shape

        if calc_flops:
            flops = self.get_flops(input, style)
        else:
            flops = 0

        # Special case for spatially-shaped style
        # Here, we early justify whether the whole feature uses the same style.
        # If that's the case, we simply use the same style, otherwise, it will use another slower logic.
        if (style is not None) and (style.ndim == 4):
            mean_style = style.mean([2,3], keepdim=True)
            is_mono_style = ((style - mean_style) < 1e-8).all()
            if is_mono_style:
                style = mean_style.squeeze()

        # Regular forward
        if style.ndim == 2:
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            # (1, ) * (1, out_ch, in_ch, k, k) * (B, 1, in_ch, 1, 1)
            # => (B, out_ch, in_ch, k, k)
            weight = self.scale * self.weight * style

            if self.demodulate:
                demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
                weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

            weight = weight.view(
                batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

            if self.upsample:
                input = input.view(1, batch * in_channel, height, width)
                weight = weight.view(
                    batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
                weight = weight.transpose(1, 2).reshape(
                    batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
                out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
                if self.no_zero_pad:
                    out = out[:, :, 1:-1, 1:-1] # Clipping head and tail, which involves zero padding
                _, _, height, width = out.shape
                out = out.view(batch, self.out_channel, height, width)
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                _, _, height, width = input.shape
                input = input.view(1, batch * in_channel, height, width)
                out = F.conv2d(input, weight, padding=self.padding, stride=2, groups=batch)
                _, _, height, width = out.shape
                out = out.view(batch, self.out_channel, height, width)

            else:
                input = input.view(1, batch * in_channel, height, width)
                out = F.conv2d(input, weight, padding=self.padding, groups=batch)
                _, _, height, width = out.shape
                out = out.view(batch, self.out_channel, height, width)
        else:
            assert (not self.training), \
                "Only accepts spatially-shaped global-latent for testing-time manipulation!"
            assert (style.ndim == 4), \
                "Only considered BxCxHxW case, but got shape {}".format(style.shape)

            # For simplicity (and laziness), we sometimes feed spatial latents 
            # that are larger than the input, center-crop for such kind of cases.
            style = self._auto_shape_align(source=style, target=input)

            # [Note]
            # Original (lossy expression):   input * (style * weight)
            # What we equivalently do here (still lossy): (input * style) * weight
            sb, sc, sh, sw = style.shape
            flat_style = style.permute(0, 2, 3, 1).reshape(-1, sc) # (BxHxW, C)
            style_mod = self.modulation(flat_style) # (BxHxW, C)
            style_mod = style_mod.view(sb, sh, sw, self.in_channel).permute(0, 3, 1, 2) # (B, C, H, W)

            input_st = (style_mod * input) # (B, C, H, W)
            weight = self.scale * self.weight

            if self.demodulate:
                # [Hubert]
                # This will be an estimation if spatilly fused styles are different.
                # In practice, the interpolation of styles do not (numerically) change drastically, so the approximation here is invisible.
                    
                """
                # This is the implementation we shown in the paper Appendix, the for-loop is slow.
                # But this version surely allocates a constant amount of memory.
                for i in range(sh):
                    for j in range(sw):
                        style_expand_s = style_mod[:, :, i, j].view(sb, 1, self.in_channel, 1, 1) # shape: (B, 1, in_ch, 1, 1)
                        simulated_weight_s = weight * style_expand_s # shape: (B, out_ch, in_ch, k, k)
                        demod_s[:, :, i, j] = torch.rsqrt(simulated_weight_s.pow(2).sum([2, 3, 4]) + 1e-8) # shape: (B, out_ch)
                """

                """
                Logically equivalent version, omits one for-loop by batching one spatial dimension.
                """
                demod = torch.zeros(sb, self.out_channel, sh, sw).to(style.device)
                for i in range(sh):
                    style_expand = style_mod[:, :, i, :].view(sb, 1, self.in_channel, sw).pow(2) # shape: (B, 1, in_ch, W)
                    weight_expand = weight.pow(2).sum([3, 4]).unsqueeze(-1) # shape: (B, out_ch, in_ch, 1)
                    simulated_weight = weight_expand * style_expand # shape: (B, out_ch, in_ch, W)
                    demod[:, :, i, :] = torch.rsqrt(simulated_weight.sum(2) + 1e-8) # shape: (B, out_ch, W)
            
                """ 
                # An even faster version that batches both height and width dimension, but allocates too much memory that is impractical in reality.
                # For instance, it allocates 40GB memory with shape (8, 512, 128, 3, 3, 31, 31).
                style_expand = style_mod.view(sb, 1, self.in_channel, 1, 1, sh, sw) # (B,      1  in_ch, 1, 1, H, W)
                weight_expand = weight.unsqueeze(5).unsqueeze(6)                    # (B, out_ch, in_ch, k, k, 1, 1)
                simulated_weight = weight_expand * style_expand # shape: (B, out_ch, in_ch, k, k, H, W)
                demod = torch.rsqrt(simulated_weight.pow(2).sum([2, 3, 4]) + 1e-8) # shape: (B, out_ch, H, W)
                """
                
                """ 
                # Just FYI. If you use the mean style over the patch, it creates blocky artifacts
                mean_style = style_mod.mean([2,3]).view(sb, 1, self.in_channel, 1, 1)
                simulated_weight_ = weight * mean_style # shape: (B, out_ch, in_ch, k, k)
                demod_ = torch.rsqrt(simulated_weight_.pow(2).sum([2, 3, 4]) + 1e-8)
                demod_ = demod_.unsqueeze(2).unsqueeze(3)
                """
 
            weight = weight.view(self.out_channel, in_channel, self.kernel_size, self.kernel_size)

            if self.upsample:
                weight = weight.transpose(0, 1).contiguous()
                out = F.conv_transpose2d(input_st, weight, padding=0, stride=2, groups=1)
                out = out[:, :, 1:-1, 1:-1] # Clipping head and tail, which involves zero padding
                _, _, height, width = out.shape
                if self.demodulate:
                    demod = F.interpolate(
                        demod,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=True)
                    out = out * demod
                out = self.blur(out)
            elif self.downsample:
                input_st = self.blur(input_st)
                out = F.conv2d(input_st, weight, padding=self.padding, stride=2, groups=1)
                if self.demodulate:
                    raise NotImplementedError("Unused, not implemented!")
                    out = out * demod
            else:
                out = F.conv2d(input_st, weight, padding=self.padding, groups=1)
                if self.demodulate and (self.padding == 0):
                    demod = demod[:, :, self.dirty_rm_size[0]:-self.dirty_rm_size[0], self.dirty_rm_size[1]:-self.dirty_rm_size[1]]
                    out = out * demod

            out = out.contiguous() # Don't know where causes discontiguity.

        return out, flops


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.testing_noise = {}

    def forward(self, image, noise=None, test_ids=None, calc_flops=False):

        # Testing with fixed noise
        if (not self.training) and (test_ids is not None):
            assert noise is None, "`test_ids` and `noise` are mutually exclusive!"

            batch, _, height, width = image.shape
            assert len(test_ids) == batch

            batch_cur_testing_noise = []
            for test_id in test_ids:
                test_id = test_id.item() if isinstance(test_id, torch.Tensor) else test_id
                if test_id not in self.testing_noise:
                    cur_testing_noise = image.new_empty(1, height, width).normal_()
                    self.testing_noise[test_id] = cur_testing_noise.cpu().detach()
                    cache_h, cache_w = height, width
                else:
                    cur_testing_noise = self.testing_noise[test_id].detach().to(image.device)
                    _, cache_h, cache_w = cur_testing_noise.shape

                    if (cache_h < height) or (cache_w < width):
                        new_testing_noise = image.new_empty(1, height, width).normal_()
                        # Replace center with old values, always assume the old noises are at the center
                        pad_h = (height - cache_h) // 2
                        pad_w = (width - cache_w) // 2
                        new_testing_noise[:, pad_h:pad_h+cache_h, pad_w:pad_w+cache_w] = cur_testing_noise
                        self.testing_noise[test_id] = new_testing_noise
                        cur_testing_noise = new_testing_noise
                        cache_h, cache_w = height, width
 
            
                # At this point, testing noise is always spatially larger or equal to current input feature
                pad_h = (cache_h - height) // 2
                pad_w = (cache_w - width) // 2 
                batch_cur_testing_noise.append(cur_testing_noise[:, pad_h:pad_h+height, pad_w:pad_w+width])
            noise = torch.stack(batch_cur_testing_noise)

        elif noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        if calc_flops:
            flops = np.prod(image.shape[1:]) * 2
        else:
            flops = 0

        output = image + self.weight * noise
        return output, flops


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch_size):
        return self.input.repeat(batch_size, 1, 1, 1)


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 2, 1],
        demodulate=True,
        no_zero_pad=False,
        disable_noise=False,
        activation="LeakyReLU",
        config=None,
        side=None,
    ):
        super().__init__()
        self.no_zero_pad = no_zero_pad
        self.upsample = upsample

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            no_zero_pad=no_zero_pad,
            config=config,
            side=side,
        )

        if disable_noise:
            self.noise = None
        else:
            self.noise = NoiseInjection()

        if activation.lower() == "LeakyReLU".lower():
            self.activate = FusedLeakyReLU(out_channel)
        else:
            raise NotImplementedError("Unknown activation {}".format(activation))

    def calc_in_spatial_size(self, out_spatial_size):
        return self.conv.calc_in_spatial_size(out_spatial_size)

    def calc_out_spatial_size(self, in_spatial_size):
        return self.conv.calc_out_spatial_size(in_spatial_size)

    def calibrate_spatial_shape(self, spatial_latent, direction, padding_mode="replicate", verbose=False, pin_loc=None):
        return self.conv.calibrate_spatial_shape(spatial_latent, direction, padding_mode=padding_mode, verbose=verbose, pin_loc=pin_loc)

    def get_noise_nch(self):
        return self.conv.out_channel

    def forward(self, input, style, noise=None, coords=None, test_ids=None, calc_flops=False):
        flops = 0
        out, cur_flops = self.conv(input, style, coords=coords, calc_flops=calc_flops)
        flops += cur_flops
        if self.noise is not None:
            out, cur_flops = self.noise(out, noise=noise, test_ids=test_ids, calc_flops=calc_flops)
            flops += cur_flops
        out = self.activate(out)
        if calc_flops:
            flops += np.prod(out.shape[1:])
        return out, flops


class ToRGB(nn.Module):
    def __init__(
        self, 
        in_channel, 
        style_dim, 
        upsample=True, 
        blur_kernel=[1, 2, 1],
        no_zero_pad=False,
        config=None,
        side=None,
    ):
        super().__init__()
        self.no_zero_pad = no_zero_pad

        if upsample:
            self.upsample = Upsample(blur_kernel, no_zero_pad=no_zero_pad)

        self.conv = ModulatedConv2d(
            in_channel=in_channel, 
            out_channel=3, 
            kernel_size=1, 
            style_dim=style_dim, 
            demodulate=False, 
            no_zero_pad=no_zero_pad,
            config=config,
            side=side)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def align_spatial_size(self, source, target):
        if source is None: return source
        _, _, cH, cW = target.shape
        _, _, sH, sW = source.shape
        if (cH == sH) and (cW == sW): return source
        # Conv has more (or even) layers, the size is always smaller than or equal to source
        assert ((sH - cH) % 2 == 0) and ((sW - cW) % 2 == 0), \
            "Should always have equal padding on two sides, got target ({}x{}) and source ({}x{})".format(cH, cW, sH, sW)
        h_st = (sH - cH) // 2
        w_st = (sW - cW) // 2
        return source[:, :, h_st:h_st+cH, w_st:w_st+cW]

    def forward(self, input, style, skip=None, coords=None, calc_flops=False):
        flops = 0
        if style.ndim == 4: # Special case that style is spatially-shaped for style fusion generation
            style = self.align_spatial_size(style, target=input)

        if coords is not None:
            coords = self.align_spatial_size(coords, target=input)
        out, cur_flops = self.conv(input, style, coords=coords, calc_flops=calc_flops)
        flops += cur_flops

        out = out + self.bias
        if calc_flops:
            flops += np.prod(out.shape[1:])

        if skip is not None:
            skip = self.upsample(skip)
            if self.no_zero_pad:
                skip = self.align_spatial_size(skip, target=out)

            out = out + skip
            if calc_flops:
                flops += np.prod(out.shape[1:])

        return out, flops

