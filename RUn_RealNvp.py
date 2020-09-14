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
    def __init__(self, mask, scale_net, train_net,couple_layers, prior):
        super(RealNvp_net, self).__init__()
        self.scale_net = nn.ModuleList([scale_net() for _ in range(couple_layers)])
        self.train_net = nn.ModuleList([train_net() for _ in range(couple_layers)])
        self.CoupleLayer = nn.ModuleList([CoupleLayer(self.scale_net[i], self.train_net[i]) for i in range(couple_layers)])
        self.couple_layers = couple_layers
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, x):
        log_det_J = x.new_zeros(x.shape[0])
        z = x
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
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.generate(z)
        return x


if __name__=="__main__":
    scale_net = lambda : nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256),
                                               nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())
    train_net = lambda : nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256),
                                               nn.LeakyReLU(), nn.Linear(256, 2))

    prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    couple_layers = 6
    mask = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))      #6 * 2
    net = RealNvp_net(mask, scale_net, train_net, couple_layers, prior)
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad==True], lr=1e-4)
    for t in range(5001):
        net.train()
        noisy_moons = datasets.make_moons(n_samples=100, noise=.05)[0].astype(np.float32)       # 100 * 2
        z, log_j = net(torch.from_numpy(noisy_moons))
        loss = net.get_loss(z, log_j)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if t % 500 == 0:
            print('iter %s:' % t, 'loss = %.3f' % loss)

    noisy_moons = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
    net.eval()
    with torch.no_grad():
        z, _ = net(torch.from_numpy(noisy_moons))
    #z = z[0].detach().numpy()
    z = z.detach().numpy()
    plt.subplot(221)
    plt.scatter(z[:, 0], z[:, 1])
    plt.title(r'$z = f(X)$')

    z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 1000)
    plt.subplot(222)
    plt.scatter(z[:, 0], z[:, 1])
    plt.title(r'$z \sim p(z)$')

    plt.subplot(223)
    x = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
    plt.scatter(x[:, 0], x[:, 1], c='r')
    plt.title(r'$X \sim p(X)$')

    plt.subplot(224)
    net.eval()
    with torch.no_grad():
        x = net.sample(1000).detach().numpy()
    plt.scatter(x[:, 0, 0], x[:, 0, 1], c='r')
    plt.title(r'$X = g(z)$')
    plt.show()
    #plt.savefig(fname='res_with_layers_256', format='jpg')


