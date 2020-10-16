import torch.nn.functional as F

from .unet_parts import *
import numpy as np


class ProbNet(nn.Module):
    def __init__(self, n_channels, feature_dim):
        super(ProbNet, self).__init__()
        self.mu = nn.Sequential(nn.Conv2d(n_channels, 64, kernel_size=1), nn.ReLU(),
                                nn.Conv2d(64, feature_dim, kernel_size=1))
        self.logvar = nn.Sequential(nn.Conv2d(n_channels, 64, kernel_size=1), nn.ReLU(),
                                    nn.Conv2d(64, feature_dim, kernel_size=1))

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class UNetProbability(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, feature_dim=6):
        super(UNetProbability, self).__init__()
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
        self.outc = OutConv(feature_dim, n_classes)

        self.prior_net = ProbNet(1, feature_dim)
        self.posterior_net = ProbNet(2, feature_dim)
        self.outc = OutConv(64 + feature_dim, n_classes)

    def forward(self, x, gt=None):
        raw_x = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.training:
            prior_mu, prior_logvar = self.prior_net(raw_x)
            posterior_mu, posterior_logvar = self.posterior_net(torch.cat([raw_x, gt], dim=1))
            posterior = self.reparameterization(posterior_mu, posterior_logvar)  # b*c*w*h
            feature = torch.cat([x, posterior], dim=1)
            logits = self.outc(feature)
            return logits, prior_mu, prior_logvar, posterior_mu, posterior_logvar

        else:
            mu, logvar = self.prior_net(raw_x)
            prior = self.reparameterization(mu, logvar)  # b*c*w*h
            feature = torch.cat([x, prior], dim=1)
            logits = self.outc(feature)
            return logits, feature

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        # sampled_z = torch.tensor(np.random.normal(0, 1, (mu.size(0), mu.size(1))))
        sampled_z = torch.normal(mu, std)
        z = sampled_z * std + mu
        return z
