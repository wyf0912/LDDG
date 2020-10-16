""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
import numpy as np


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        feature_map = x
        logits = self.outc(feature_map)
        return logits, feature_map


def kl_gaussian(Dx_m, Dx_var):
    kl_loss = -0.5 * torch.sum(1 + Dx_var - Dx_m.pow(2) - Dx_var.exp())
    return kl_loss


class UNetOurs(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, feature_dim=8):
        super(UNetOurs, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_dim = feature_dim

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.mu = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                nn.Conv2d(64, feature_dim, kernel_size=1))
        self.logvar = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                    nn.Conv2d(64, feature_dim, kernel_size=1))
        self.outc = OutConv(feature_dim, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        if self.training:
            feature = self.reparameterization(mu, logvar)  # b*c*w*h
            logits = self.outc(feature)
            return logits, torch.stack([mu, logvar, feature], dim=1)
        else:
            feature = mu
            logits = self.outc(feature)
            return logits, mu

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        # sampled_z = torch.tensor(np.random.normal(0, 1, (mu.size(0), mu.size(1))))
        sampled_z = torch.normal(mu, std)
        z = sampled_z * std + mu
        return z


class MetricNet(nn.Module):
    def __init__(self, in_channel=64):
        super(MetricNet, self).__init__()
        self.fc1 = nn.Linear(in_channel, 1024)
        self.fc2 = nn.Linear(1024, 256)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        return x
