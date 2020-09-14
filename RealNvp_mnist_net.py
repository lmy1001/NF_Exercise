import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
import torch.optim as optim
from torch import distributions
import matplotlib.pyplot as plt

class CoupleLayer(nn.Module):
    def __init__(self, scale_net, train_net):
        super(CoupleLayer, self).__init__()
        self.scale_net = scale_net
        self.train_net = train_net

    def forward(self, z, mask):
        z_ = mask * z
        s = self.scale_net(z_) * (1 - mask)
        t = self.train_net(z_) * (1 - mask)
        z = (1 - mask) * (z - t) * torch.exp(-s) + z_
        return z, s



class RealNvp_net(nn.Module):
    def __init__(self, scale_net, train_net,couple_layers, prior, input_size):
        super(RealNvp_net, self).__init__()
        mask1 = torch.arange(input_size * input_size) % 2
        mask2 = 1 - mask1
        mask = []
        for i in range( int(couple_layers / 2)):
            mask.append(mask1)
            mask.append(mask2)
        mask = torch.cat(mask, dim=0)
        self.mask = mask.view(couple_layers, -1)

        self.scale_net = nn.ModuleList([scale_net() for _ in range(couple_layers)])
        self.train_net = nn.ModuleList([train_net() for _ in range(couple_layers)])
        self.CoupleLayer = nn.ModuleList([CoupleLayer(self.scale_net[i], self.train_net[i]) for i in range(couple_layers)])
        self.couple_layers = couple_layers
        self.prior = prior

    def forward(self, x):
        log_det_J = x.new_zeros(x.shape[0])
        z = x.view(x.shape[0], -1)
        for i in reversed(range(self.couple_layers)):
            z, s = self.CoupleLayer[i](z, self.mask[i])
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def get_loss(self, z, log_det_J):
        loss = self.prior.log_prob(z) + log_det_J
        loss = -loss.mean()
        return loss

    def generate(self, z):
        x = z
        for i in range(self.couple_layers):
            x_ = x * self.mask[i]
            s = self.scale_net[i](x_) * (1 - self.mask[i])
            t = self.train_net[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1)).squeeze()
        logp = self.prior.log_prob(z)
        x = self.generate(z)
        return x



