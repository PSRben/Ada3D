import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.init import calculate_gain
from model.hisr.adaptive3d import Adaptive3D


class MultiEdge(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.laplace_kernel = torch.tensor([[0, 1, 0],
                                            [1, -4, 1],
                                            [0, 1, 0]],
                                           dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).to(self.device)
        self.sobel_kernel_x = torch.tensor([[-1, -2, -1],
                                            [0, 0, 0],
                                            [1, 2, 1]],
                                           dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).to(self.device)
        self.sobel_kernel_y = torch.tensor([[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]],
                                           dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).to(self.device)
        self.prewitt_kernel_x = torch.tensor([[-1, -1, -1],
                                              [0, 0, 0],
                                              [1, 1, 1]],
                                             dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).to(self.device)
        self.prewitt_kernel_y = torch.tensor([[-1, 0, 1],
                                              [-1, 0, 1],
                                              [-1, 0, 1]],
                                             dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).to(self.device)
        self.scharr_kernel_x = torch.tensor([[-3, -10, -3],
                                             [0, 0, 0],
                                             [3, 10, 3]],
                                            dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).to(self.device)
        self.scharr_kernel_y = torch.tensor([[-3, 0, 3],
                                             [-10, 0, 10],
                                             [3, 0, 3]],
                                            dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).to(self.device)

    def forward(self, x):
        # x: (B, 1, H, W)
        b, c, h, w = x.shape
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        laplace = F.conv2d(x_pad, self.laplace_kernel.repeat(c, 1, 1, 1), stride=1, padding=0, groups=c)
        sobel_x = F.conv2d(x_pad, self.sobel_kernel_x.repeat(c, 1, 1, 1), stride=1, padding=0, groups=c)
        sobel_y = F.conv2d(x_pad, self.sobel_kernel_y.repeat(c, 1, 1, 1), stride=1, padding=0, groups=c)
        sobel = 0.5 * torch.abs(sobel_x) + 0.5 * torch.abs(sobel_y)
        prewitt_x = F.conv2d(x_pad, self.prewitt_kernel_x.repeat(c, 1, 1, 1), stride=1, padding=0, groups=c)
        prewitt_y = F.conv2d(x_pad, self.prewitt_kernel_y.repeat(c, 1, 1, 1), stride=1, padding=0, groups=c)
        prewitt = 0.5 * torch.abs(prewitt_x) + 0.5 * torch.abs(prewitt_y)
        scharr_x = F.conv2d(x_pad, self.scharr_kernel_x.repeat(c, 1, 1, 1), stride=1, padding=0, groups=c)
        scharr_y = F.conv2d(x_pad, self.scharr_kernel_y.repeat(c, 1, 1, 1), stride=1, padding=0, groups=c)
        scharr = 0.5 * torch.abs(scharr_x) + 0.5 * torch.abs(scharr_y)

        pan_x = torch.zeros([b, c, h, w], dtype=torch.float, requires_grad=False).to(self.device)
        pan_x[:, :, 1: h, :] = x[:, :, 0: h-1, :]
        pan_x[:, :, 0, :] = x[:, :, 0, :]
        pan_x = torch.abs(pan_x - x)
        pan_y = torch.zeros([b, c, h, w], dtype=torch.float, requires_grad=False).to(self.device)
        pan_y[:, :, :, 1: w] = x[:, :, :, 0: w-1]
        pan_y[:, :, :, 0] = x[:, :, :, 0]
        pan_y = torch.abs(pan_y - x)

        return torch.cat([pan_x, pan_y, laplace, sobel, prewitt, scharr], dim=1)


class PixelShuffle(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.upsamle = nn.Sequential(
            nn.Conv2d(dim, dim*(scale**2), 1, 1, 0, bias=False),
            nn.Conv2d(dim*(scale**2), dim*(scale**2), 3, 1, 1, bias=False, groups=dim*(scale**2)),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        return self.upsamle(x)


class ResBlock2D(nn.Module):
    def __init__(self, dim, res_se_ratio):
        super().__init__()
        hidden_dim = int(res_se_ratio * dim)
        self.conv0 = nn.Conv2d(dim, hidden_dim, 3, 1, 1)
        self.conv1 = nn.Conv2d(hidden_dim, dim, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv0(x))
        rs1 = self.conv1(rs1)
        rs = torch.add(x, rs1)
        return rs


class ResBlock3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 3, 1, 1)
        self.conv1 = nn.Conv3d(dim, dim, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv0(x))
        rs1 = self.conv1(rs1)
        rs = torch.add(x, rs1)
        return rs


class ResBlockAda3D(nn.Module):
    def __init__(self,
                 spe_channel,
                 spa_channel,
                 spectral_dim,
                 spatial_dim,
                 kernel_size,
                 spe_se_ratio,
                 spa_se_ratio,
                 channel_se_ratio
                 ):
        super().__init__()
        self.conv0 = Adaptive3D(spe_channel, spa_channel, spectral_dim, spatial_dim, 
                                kernel_size, spe_se_ratio, spa_se_ratio, channel_se_ratio)
        self.conv1 = Adaptive3D(spe_channel, spa_channel, spectral_dim, spatial_dim, 
                                kernel_size, spe_se_ratio, spa_se_ratio, channel_se_ratio)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, y):
        rs1 = self.relu(self.conv0(x, y))
        rs1 = self.conv1(rs1, y)
        rs = torch.add(x, rs1)
        return rs


class ConvNet3D(nn.Module):
    def __init__(self, 
                 spe_channel=32,
                 spa_channel=64,
                 spectral_dim=31,
                 spatial_dim=3,
                 layer_num=4,
                 scale=4,
                 kernel_size=3,
                 spe_se_ratio=0.5,
                 spa_se_ratio=0.375,
                 channel_se_ratio=0.125,
                 res_se_ratio=1,
                 norm=True,
                 device='cuda',
                 edge=False
                 ):
        super().__init__()
        self.spe_channel = spe_channel
        self.spa_channel = spa_channel
        self.spectral_dim = spectral_dim
        self.spatial_dim = spatial_dim
        self.layer_num = layer_num
        self.scale = scale
        self.kernel_size = kernel_size
        self.spe_se_ratio = spe_se_ratio
        self.spa_se_ratio = spa_se_ratio
        self.channel_se_ratio = channel_se_ratio
        self.res_se_ratio = res_se_ratio
        self.norm = norm
        self.device = device
        self.edge = edge
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        if self.edge is True:
            self.edge_detection = MultiEdge(self.device)
            self.raise_spa_dim = nn.Sequential(
                nn.Conv2d(7 * self.spatial_dim + self.spectral_dim, self.spa_channel, 3, 1, 1),
                self.relu
            )
        else:
            self.raise_spa_dim = nn.Sequential(
                nn.Conv2d(self.spatial_dim + self.spectral_dim, self.spa_channel, 3, 1, 1),
                self.relu
            )

        self.raise_spe_dim = nn.Sequential(
            nn.Conv3d(1, self.spe_channel, 3, 1, 1),
            self.relu
        )
        self.retore_dim = nn.Conv3d(self.spe_channel, 1, 3, 1, 1)

        self.spectral_layers = nn.ModuleList([])
        self.spatial_layers = nn.ModuleList([])
        for _ in range(self.layer_num):
            self.spatial_layers.append(ResBlock2D(self.spa_channel, self.res_se_ratio))
            self.spectral_layers.append(ResBlockAda3D(self.spe_channel,
                                                      self.spa_channel,
                                                      self.spectral_dim,
                                                      self.spatial_dim,
                                                      self.kernel_size,
                                                      self.spe_se_ratio,
                                                      self.spa_se_ratio,
                                                      self.channel_se_ratio))

    def forward(self, x, y):
        # x: (BxLxhxw)
        # y: (Bx1xHxW)
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        skip = x
        if self.edge is True:
            multi_edge = self.edge_detection(y)
            y = self.raise_spa_dim(torch.cat([x, y, multi_edge], dim=1))
        else:
            y = self.raise_spa_dim(torch.cat([x, y], dim=1))
        x = self.raise_spe_dim(x.unsqueeze(1))
        for spe_layer, spa_layer in zip(self.spectral_layers, self.spatial_layers):
            y = spa_layer(y)
            x = spe_layer(x, y)
        return self.retore_dim(x).squeeze(1) + skip