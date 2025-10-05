import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.init import calculate_gain


class Norm(nn.Module):
    def __init__(self, in_channels, kernel_size, filter_type,
                 nonlinearity='leaky_relu', running_std=False, running_mean=False):
        assert filter_type in ('spatial', 'spectral', 'none')
        assert in_channels >= 1
        super(Norm, self).__init__()
        self.in_channels = in_channels
        self.filter_type = filter_type
        self.runing_std = running_std
        self.runing_mean = running_mean
        std = calculate_gain(nonlinearity) / kernel_size
        if running_std:
            self.std = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 3) * std, requires_grad=True)
        else:
            self.std = std
        if running_mean:
            self.mean = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 3), requires_grad=True)

    def forward(self, x):
        if self.filter_type == 'spatial':
            b, _, h, w = x.size()
            x = x.reshape(b, self.in_channels, -1, h, w)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-10)
            x = x.reshape(b, _, h, w)
            if self.runing_std:
                x = x * self.std[None, :, None, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None, None]

        elif self.filter_type == 'spectral':
            b, _, l = x.size()
            x = x.reshape(b, self.in_channels, -1, l)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, l)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, l) + 1e-10)
            x = x.reshape(b, _, l)
            if self.runing_std:
                x = x * self.std[None, :, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None]

        elif self.filter_type == 'none':
            x = x

        else:
            raise RuntimeError('Unsupported filter type {}'.format(self.filter_type))
        return x


class CreateSpectralKernel(nn.Module):
    def __init__(self, dim,
                 spectral_dim=8,
                 kernel_size=3,
                 spe_se_ratio=2,
                 norm=True):
        super().__init__()
        self.dim = dim
        self.spectral_dim = spectral_dim
        self.kernel_size = kernel_size
        self.spe_se_ratio = spe_se_ratio
        if norm is True:
            self.norm_type = 'spectral'
        else:
            self.norm_type = 'none'

        self.out_dim = self.dim * (self.kernel_size ** 3)
        self.hidden_dim = int(self.dim * self.spe_se_ratio)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.attention = nn.Sequential(
            nn.Conv3d(self.dim, self.hidden_dim, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(self.hidden_dim, self.out_dim, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        )
        self.norm = Norm(self.dim, self.kernel_size, self.norm_type, 'leaky_relu')

    def forward(self, x):
        # input: (B, C, L, H, W)
        # output: (B, C, L, 1, 1, k, k, k)
        b, c, l, h, w = x.shape
        k = self.kernel_size
        x = x.reshape(b, c*l, h, w)
        x = self.pooling(x)  # (B, CxL, 1, 1)
        x = x.reshape(b, c, l, 1, 1)
        x = self.attention(x)  # (B, Cxk^3, L, 1, 1)
        x = x.reshape(b, self.out_dim, l)  # (B, Cxk^3, L)
        x = self.norm(x)  # (B, Cxk^3, L)
        x = x.reshape(b, c, 1, 1, k, k, k, l).permute(0, 1, 7, 2, 3, 4, 5, 6)
        return x


class CreateSpatialKernel(nn.Module):
    def __init__(self, dim,
                 spatial_dim=1,
                 kernel_size=3,
                 spa_se_ratio=2,
                 norm=True):
        super().__init__()
        self.dim = dim
        self.spatial_dim = spatial_dim
        self.kernel_size = kernel_size
        self.spa_se_ratio = spa_se_ratio
        if norm is True:
            self.norm_type = 'spatial'
        else:
            self.norm_type = 'none'

        self.out_dim = self.dim * (self.kernel_size ** 3)
        self.hidden_dim = int(self.dim * self.spa_se_ratio)

        self.attention = nn.Sequential(
            nn.Conv2d(self.dim, self.hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_dim, self.out_dim, 3, 1, 1)
        )
        self.norm = Norm(self.dim, self.kernel_size, self.norm_type, 'leaky_relu')

    def forward(self, x):
        # input: (B, C, H, W)
        # output: (B, C, 1, H, W, k, k, k)
        b, c, h, w = x.shape
        k = self.kernel_size
        x = self.attention(x)  # (B, Cxk^3, H, W)
        x = self.norm(x)  # (B, Cxk^3, H, W)
        x = x.reshape(b, c, 1, k, k, k, h, w).permute(0, 1, 2, 6, 7, 3, 4, 5)
        return x


class CreateAdaBias(nn.Module):
    def __init__(self, dim,
                 spectral_dim=8,
                 kernel_size=3,
                 channel_se_ratio=0.5,
                 norm=True):
        super().__init__()
        self.dim = dim
        self.spectral_dim = spectral_dim
        self.kernel_size = kernel_size
        self.channel_se_ratio = channel_se_ratio
        if norm is True:
            self.norm_type = 'bias'
        else:
            self.norm_type = 'none'

        self.in_dim = self.dim * 2
        self.out_dim = self.dim
        self.hidden_dim = int(self.dim * self.channel_se_ratio)

        self.pooling2d = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.create_spatial = nn.Conv2d(self.dim, 1, 3, 1, 1)
        self.create_spectral = nn.Conv3d(self.dim, 1, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        self.create_channel = nn.Sequential(
            nn.Conv3d(self.in_dim, self.hidden_dim, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(self.hidden_dim, self.out_dim, 1, 1, 0)
        )

    def forward(self, x, y):
        # x: (B, C, L, H, W)
        # y: (B, C, H, W)
        # output: (B, 1, L, H, W)
        b, c, l, h, w = x.shape
        # create spectral bias
        spectral_bias = x.reshape(b, c * l, h, w)
        spectral_bias = self.pooling2d(spectral_bias)  # (B, C*L, 1, 1)
        spectral_bias = spectral_bias.reshape(b, c, l, 1, 1)
        spectral_bias = self.create_spectral(spectral_bias)  # (B, 1, L, 1, 1)
        # create spatial bias
        spatial_bias = self.create_spatial(y)  # (B, 1, H, W)
        # create channel bias = normal bias
        spe_channel = self.pooling3d(x)  # (B, C, 1, 1, 1)
        spa_channel = self.pooling2d(y)  # (B, C, 1, 1)
        channel = torch.cat([spe_channel, spa_channel.unsqueeze(2)], dim=1)  # (B, 2C, 1, 1, 1)
        channel_bias = self.create_channel(channel)  # (B, C, 1, 1, 1)
        # combine
        content_bias = torch.mul(spectral_bias, spatial_bias.unsqueeze(1))  # (B, 1, L, H, W)
        ada_bias = torch.mul(content_bias, channel_bias)  # (B, C, L, H, W)
        return ada_bias


class Adaptive3D(nn.Module):
    def __init__(self, dim,
                 spectral_dim=8,
                 spatial_dim=1,
                 kernel_size=3,
                 spe_se_ratio=0.5,
                 spa_se_ratio=0.5,
                 channel_se_ratio=0.5,
                 norm=True):
        super().__init__()
        self.dim = dim
        self.spectral_dim = spectral_dim
        self.spatial_dim = spatial_dim
        self.kernel_size = kernel_size
        self.spe_se_ratio = spe_se_ratio
        self.spa_se_ratio = spa_se_ratio
        self.channel_se_ratio = channel_se_ratio
        self.norm = norm

        self.spectral_kernel = CreateSpectralKernel(self.dim,
                                                    self.spectral_dim,
                                                    self.kernel_size,
                                                    self.spe_se_ratio,
                                                    self.norm)
        self.spatial_kernel = CreateSpatialKernel(self.dim,
                                                  self.spatial_dim,
                                                  self.kernel_size,
                                                  self.spa_se_ratio,
                                                  self.norm)
        self.bias = CreateAdaBias(self.dim,
                                  self.spectral_dim,
                                  self.kernel_size,
                                  self.channel_se_ratio,
                                  self.norm)

    def adaptive_3d_conv(self, x):
        k = self.kernel_size
        pad = k // 2
        x_pad = F.pad(x, (pad, pad, pad, pad, pad, pad), mode='constant', value=0)
        x_unfold = ((x_pad.unfold(-3, k, pad)).unfold(-3, k, pad)).unfold(-3, k, pad)
        return torch.sum(torch.mul(x_unfold, self.adaptive3d_kernel), [-3, -2, -1])

    def forward(self, x, y):
        # x (B, C, L, H, W): spectral input
        # y (B, C, H, W): spatial input
        # output: (B, C, L, H, W)
        spectral_kernel = self.spectral_kernel(x)
        spatial_kernel = self.spatial_kernel(y)
        ada_bias = self.bias(x, y)
        self.adaptive3d_kernel = torch.mul(spectral_kernel, spatial_kernel)
        output = self.adaptive_3d_conv(x) + ada_bias
        return output
