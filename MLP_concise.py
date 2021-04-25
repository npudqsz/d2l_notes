# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:36:19 2021

@author: qdu
"""
import IPython

import torch
from torch import nn
from torch.utils import data

from torchvision import datasets
from torchvision import transforms

def get_workers():
    """ number of workers for DataLoader """
    return 4

def init_weight_bias(m):
    """ initialize linear layers' weight and bias """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
        
def accuracy(y_hat, y):
    """ accuracy of predicted labels (proba) 'y_hat' in terms of real labels 'y';
        return the number of correct labels.
    """
    indices = torch.argmax(y_hat, axis=1, keepdim=True)
    cmp = indices.type(y.dtype) == y.reshape(indices.shape)
    return sum(cmp)

def eval_accuray(net, data_iter):
    """ evaluate accuracy of model 'net' in dataset loader 'data_iter' """
    if isinstance(net, nn.Module):
        net.eval()      # set model to evaluation mode
    num_correct, num_all = 0, 0
    for X, y in data_iter:
        num_correct += accuracy(net(X), y)
        num_all += y.numel()
    return float(num_correct / num_all)

def get_fashion_mnist_labels(labels):
    """ get real labels text from their numbers of Fashion-MNIST dataset """
    text = {
        '0' : 't-shirt',
        '1' : 'trouser',
        '2' : 'pullover',
        '3' : 'dress',
        '4' : 'coat',
        '5' : 'sandal',
        '6' : 'shirt',
        '7' : 'sneaker',
        '8' : 'bag',
        '9' : 'ankle boot'
    }
    return [text[str(int(i))] for i in labels]

if __name__ == '__main__':
    # IPython line magic
    IPython.get_ipython().run_line_magic('matplotlib', 'inline')
    
    #%% dataset of FashionMNIST
    trans = transforms.ToTensor()
    mnist_train = datasets.FashionMNIST(root='../data', train=True,
                                        transform=trans, download=True)
    mnist_test  = datasets.FashionMNIST(root='../data', train=True,
                                        transform=trans, download=True)
    
    #%% dataloader
    batch_size = 200
    train_iter = data.DataLoader(mnist_train, batch_size,
                                 shuffle=True, num_workers=get_workers())
    test_iter  = data.DataLoader(mnist_test,  batch_size,
                                 shuffle=True, num_workers=get_workers())
    
    #%% neural networks (Multi-Layers Perceptron)
    in_features, hidden_features, out_features = 28*28, 2**8, 10
    # MLP (two layers)
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(in_features, hidden_features),
                        nn.ReLU(),
                        nn.Linear(hidden_features, out_features))
    # # softmax (one layer)
    # net = nn.Sequential(nn.Flatten(),
    #                     nn.Linear(in_features, out_features))
    net.apply(init_weight_bias)
    
    #%% loss function
    # input: y_hat before normalization (softmax calculation is done inside cross entropy function)
    loss = nn.CrossEntropyLoss()
    
    #%% optimizer
    lr = 0.1
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    
    #%% training
    num_epochs = 10
    for epoch in range(num_epochs):
        if isinstance(net, nn.Module):
            net.train()         # set model to training mode
        num_loss, num_correct, num_all = 0, 0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            num_loss += l * y.numel()
            num_correct += accuracy(y_hat, y)
            num_all += y.numel()
        loss_train = float(num_loss / num_all)      # loss for traing dataset
        acc_train = float(num_correct / num_all)    # accuracy for training dataset
        acc_test = eval_accuray(net, test_iter)     # accuracy for test dataset
        print(f'epoch : {epoch}, train loss : {loss_train:.3f}, ' + 
              f'train acc : {acc_train:.3f}, test acc : {acc_test:.3f}')
        
    #%% prediction
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))