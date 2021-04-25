# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:54:40 2021

@author: qdu
"""
import IPython

import torch
from torch import nn
from torch.utils import data

import matplotlib.pyplot as plt

def get_data(W, b, sample_size):
    """ Gaussian-noised dataset of linear model y = XW + b """
    num_in_features = len(W)
    X = torch.rand(sample_size, num_in_features)
    y = torch.matmul(X, W) + b
    y += torch.normal(0.0, 0.4, y.shape)
    return X, y.reshape(-1,1)       # reshape as input to net() should be (-1,len(W))

# @torch.no_grad()
def init_weight(m):
    if type(m) == nn.Linear:
        # m.weight.fill_(1.0)
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
    
if __name__ == '__main__':
    # IPython line magic => %matplotlib inline
    IPython.get_ipython().run_line_magic('matplotlib', 'inline')
    
    #%% data & dataset of linear model y = XW + b
    W = torch.tensor([3.0, 5.0]).reshape(-1,1)
    b = torch.tensor([4.5]).reshape(-1,1)
    size = 1000
    features, labels = get_data(W, b, size)
    dataset = data.TensorDataset(features, labels)
    
    #%% dataloader
    batch_size = 20
    data_iter = data.DataLoader(dataset = dataset, batch_size = batch_size,
                                shuffle = True, num_workers = 1)
    
    #%% neural network
    net = nn.Sequential(nn.Linear(in_features = len(W), out_features = len(b)))
    net.apply(init_weight)
    
    #%% loss function
    loss = nn.MSELoss()
    
    #%% optimizer
    lr = 0.1   # learning rate
    trainer = torch.optim.SGD(net.parameters(), lr = lr)
    
    #%% training
    num_epochs = 10
    for epoch in range(num_epochs):
        for x, y in data_iter:
            l = loss(net(x), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch} : loss {l:.3f}')
        
    #%% result
    # 2D scatter
    if len(W) == 1:
        plt.scatter(dataset.tensors[0].numpy(), dataset.tensors[1].numpy())
        w_train = net[0].weight.data
        b_train = net[0].bias.data
        x = torch.linspace(0,1,50).reshape(-1,1)
        y = w_train * x + b_train
        plt.plot(x.numpy(), y.numpy(), '--r')
        
    # 3D scatter
    elif len(W) == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(features.numpy()[:,0], features.numpy()[:,1], labels.numpy())
        w1_train, w2_train = net[0].weight.data.reshape(-1,1)
        b_train = net[0].bias.data
        x1 = torch.linspace(0,1,50)
        x2 = torch.linspace(0,1,50)
        X1, X2 = torch.meshgrid(x1, x2)
        y = w1_train * X1 + w2_train * X2 + b_train
        ax.plot_surface(X1.numpy(), X2.numpy(), y.numpy(), cmap='gray', shade=True)
        
    W_train = net[0].weight.data
    b_train = net[0].bias.data
    print(f'W_true  : {W.reshape(1,-1)}')
    print(f'W_train : {W_train.reshape(1,-1)}')
    print(f'b_true  : {b.reshape(1,-1)}')
    print(f'b_train : {b_train.reshape(1,-1)}')
    