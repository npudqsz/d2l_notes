# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 22:04:44 2021

@author: qdu
"""
import IPython
from IPython import display

import timeit
import math
import numpy as np

import torch
from torch import nn
from torch.utils import data

import matplotlib.pyplot as plt

def eval_loss(net, data_iter, loss):
    """ evaluate loss of model 'net' in dataloader 'data_iter' """
    num_loss, num_all = 0.0, 0.0
    for X, y in data_iter:
        l = loss(net(X), y)
        num_loss += l
        num_all += l.numel()
    return float(num_loss / num_all)

class Animator:
    """ plot curve evolution in real time """
    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None,
                 xscale='linear', yscale='linear', legend=None):
        display.set_matplotlib_formats('svg')       # 'svg' makes plot more clear
        fig = plt.figure(figsize=(3.5,2.5))
        axis = fig.add_subplot(111)
        self.ca = lambda : self.config_axis(xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.fig, self.axis, self.X, self.Y, self.legend = fig, axis, None, None, legend
        
    def config_axis(self, xlabel=None, ylabel=None, xlim=None, ylim=None,
                 xscale='linear', yscale='linear', legend=None):
        axis = self.axis
        if xlabel is not None:
            axis.set_xlabel(xlabel)
        if ylabel is not None:
            axis.set_ylabel(ylabel)
        if xlim is not None:
            axis.set_xlim(*xlim)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.set_xscale(xscale)
        axis.set_yscale(yscale)
        if legend is None:
            legend = []
        else:
            axis.legend(legend)
        axis.grid()
        
    def add(self, x, y):
        if not hasattr(y, '__len__'):
            y = [y]     # in case that y is a value and not a list, convert the value to a list
        num_curve = len(y)      # number of curve
        if not hasattr(x, '__len__'):
            x = [x] * num_curve
        if not self.X:
            self.X = [[] for _ in range(num_curve)]  # list of list for recording different curves' values
        if not self.Y:
            self.Y = [[] for _ in range(num_curve)]
            
        for i, (a,b) in enumerate(zip(x, y)):
            self.X[i].append(a)
            self.Y[i].append(b)

        axis = self.axis
        axis.cla()
        for i in range(num_curve):
            axis.plot(self.X[i], self.Y[i])
        self.ca()
        display.display(self.fig)
        display.clear_output(wait=True)
        
def get_workers():
    """ number of workers for DataLoader """
    return 0
        
def training(num_epochs):
    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[0,num_epochs-1],
                        ylim=[1e-3, 1e2], yscale='log', legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        loss_train = eval_loss(net, train_iter, loss)
        loss_test  = eval_loss(net, test_iter, loss)
        if epoch==0 or (epoch+1)%20==0:     # this plot step can be modified to save time
            animator.add(epoch, (loss_train, loss_test))
        print(f'epoch : {epoch}, loss_train : {loss_train:.3f}, loss_test : {loss_test:.3f}')
        str_pred = ['%.3f' %i for i in net[0].weight.data[0]]
        str_true = ['%.3f' %i for i in true_w[:4]]
        print('weight_pred : ', ' '.join(str_pred))
        print('weight_true : ', ' '.join(str_true), '\n')

if __name__ == '__main__':
    IPython.get_ipython().run_line_magic('matplotlib', 'inline')
    
    # computation device : 'cpu' or 'gpu'/'cuda'; 'cuda' can be sometimes slower than 'cpu'..
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    dev = torch.device('cpu')
    
    max_degree = 20
    n_train, n_test = 100, 100
    true_w = np.zeros(max_degree)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
    
    #%% features & labels
    features = np.random.normal(0, 1, (n_train+n_test, 1))
    np.random.shuffle(features)
    poly_features = np.power(features, np.arange(max_degree).reshape(1,-1))
    for i in range(max_degree):
        poly_features[:,i] /= math.gamma(i+1)     # gamma(n) = (n-1)!
    labels = np.dot(poly_features, true_w).reshape(-1,1)
    labels += np.random.normal(0, 0.1, labels.shape)
    
    poly_features = torch.tensor(poly_features, dtype=torch.float32, device=dev)
    labels = torch.tensor(labels, dtype=torch.float32, device=dev)
    
    #%% dataset
    n_degree = 4    # underfitting / overfitting depending on its value in terms of max_degree
    dataset_train = data.TensorDataset(poly_features[:n_train, :n_degree], labels[:n_train])
    dataset_test  = data.TensorDataset(poly_features[n_train:, :n_degree], labels[n_train:])
    
    #%% dataloader
    batch_size = 10
    train_iter = data.DataLoader(dataset_train, batch_size,
                                 shuffle=True, num_workers=get_workers())
    test_iter  = data.DataLoader(dataset_test,  batch_size,
                                 shuffle=True, num_workers=get_workers())
    
    #%% neural network
    in_features = n_degree
    out_features = labels.shape[-1]
    net = nn.Sequential(nn.Linear(in_features, out_features, bias=False).to(dev))
    
    #%% loss function
    loss = nn.MSELoss()
    
    #%% optimizer
    lr = 0.01
    trainer = torch.optim.SGD(net.parameters(), lr)
    
    #%% training & timing
    num_epochs = 1500
    t0 = timeit.Timer(
        stmt='training(x)',
        setup='from __main__ import training',
        globals={'x':num_epochs})
    print(f'training time : {t0.timeit(1):.3f} s')
    
