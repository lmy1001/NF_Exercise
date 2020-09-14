import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import datasets
from torch import distributions
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from RealNvp_mnist_net import RealNvp_net

transform = transforms.Compose(
[transforms.ToTensor,
transforms.Normalize((0.1307),(0.3081))
])

batch_size = 128

#Data set
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
#image_size: 128 * 1 * 28 * 28
#labels_size: 128
print(len(train_loader), len(test_loader))          #train_loader: 469, test_loader: 79 batches

scale_net = lambda : nn.Sequential(nn.Linear(28 * 28, 128), nn.Tanh(), nn.Linear(128, 256),nn.Tanh(),
                                   nn.Linear(256, 512), nn.Tanh(), nn.Linear(512, 1024), nn.Tanh(),
                                   nn.Linear(1024, 1024), nn.Tanh(), nn.Linear(1024, 28 * 28), nn.Tanh())
train_net = lambda : nn.Sequential(nn.Linear(28 * 28, 128), nn.Tanh(), nn.Linear(128, 256), nn.Tanh(),
                                   nn.Linear(256, 512), nn.Tanh(), nn.Linear(512, 1024), nn.Tanh(),
                                   nn.Linear(1024, 1024), nn.Tanh(), nn.Linear(1024, 28 * 28), nn.LeakyReLU())

couple_layers = 4
input_size = train_dataset[0][0].shape[1]
sample_size = input_size * (train_dataset[0][0].shape[2])

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
prior = distributions.MultivariateNormal(torch.zeros(sample_size), torch.eye(sample_size))
net = RealNvp_net(scale_net, train_net, couple_layers, prior, input_size)
net = net.to(device)
optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad == True], lr=1e-4, weight_decay=1e-6)

train_writer = SummaryWriter(logdir='./log/train')
val_writer = SummaryWriter(logdir='./log/val')
for t in range(250):
    net.train()
    for i, (images,labels) in enumerate(train_loader, 0):
        input = images
        input = input.to(device)
        z, log_j = net(input)
        loss = net.get_loss(z, log_j)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        batch_count = (t - 1) * len(train_loader) + i + 1
        train_writer.add_scalar('batch_loss', loss.item(), batch_count)
        if i % 10 == 0:
            net.eval()
            _, (test_images, labels) = next(enumerate(test_loader, 0))
            test_input = test_images
            test_input = test_input.to(device)
            z_test, log_j_test = net(test_input)
            test_loss = net.get_loss(z_test, log_j_test)
            val_writer.add_scalar('batch_loss', test_loss, batch_count)
    samples = net.sample(batch_size)
    samples = samples.view(samples.shape[0], input_size, input_size)
    samples = torch.unsqueeze(samples, 1)
    save_image(samples, './res/sample_' + str(t) + '.png')
    print('iter %s:' % t, 'loss = %.3f' % loss, 'eval_loss = %.3f' % test_loss)






