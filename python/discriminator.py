# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:22:35 2018
@author: natnij

Based on SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient, 
    Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu.
    Paper available here: https://arxiv.org/abs/1609.05473
Translated from the original tensorflow repo: 
    https://github.com/LantaoYu/SeqGAN, and adjusted for wider usability.
Many thanks to the original authors.
"""
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (SEQ_LENGTH,EMB_SIZE,FILTER_SIZE,NUM_FILTER,
                    DIS_NUM_EPOCH_PRETRAIN,DEVICE,openLog)
from data_processing import gen_record,gen_label

class Highway(nn.Module):
    def __init__(self, in_features, out_features, num_layers=1, bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.bias = bias
        self.cells = nn.ModuleList()
        for idx in range(self.num_layers):
            g = nn.Sequential(
                    nn.Linear(self.in_features, self.out_features),
                    nn.ReLU(inplace=True)
                    )
            t = nn.Sequential(
                    nn.Linear(self.in_features, self.out_features),
                    nn.Sigmoid()
                    )
            self.cells.append(g)
            self.cells.append(t)
        
    def forward(self, x):
        for i in range(0,len(self.cells),2):
            g = self.cells[i]
            t = self.cells[i+1]
            nonlinearity = g(x)
            transformGate = t(x) + self.bias
            x = nonlinearity * transformGate + (1-transformGate) * x
        return x        

class Discriminator(nn.Module):
    def __init__(self, filter_size=None, num_filter=None, dropoutRate=0.0, vocab_size=10):
        super().__init__()
        if filter_size is None:
            self.filter_size = [SEQ_LENGTH]
        else:
            assert isinstance(filter_size, list)
            self.filter_size = filter_size.copy()
        if num_filter is None:
            self.num_filter = [100]
        else:
            assert len(filter_size)==len(num_filter)
            self.num_filter = num_filter.copy()
        self.num_filter_total = sum(self.num_filter)
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, EMB_SIZE)
        self.convs = nn.ModuleList()
        for fsize, fnum in zip(self.filter_size, self.num_filter):
            # kernel_size = depth, height, width
            conv = nn.Sequential(
                    nn.Conv2d(in_channels=1,out_channels=fnum,
                             kernel_size=(fsize,EMB_SIZE),
                             padding=0,stride=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(SEQ_LENGTH-fsize+1,1),stride=1)
                    )
            self.convs.append(conv)
        
        self.highway = Highway(self.num_filter_total,self.num_filter_total,
                               num_layers=1, bias=0)
        self.dropout = nn.Dropout(p=dropoutRate)
        self.fc = nn.Linear(sum(self.num_filter),2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embeds = self.embedding(x.long())
        xs = list()
        for i,conv in enumerate(self.convs):
            x0 = conv(embeds.view(-1,1,SEQ_LENGTH,EMB_SIZE))
            x0 = x0.view((x0.shape[0],x0.shape[1]))
            xs.append(x0)
        cats = torch.cat(xs,1)
        highway = self.highway(cats)
        dropout = F.relu(self.dropout(highway))
        fc = F.relu(self.fc(dropout))
        y_prob = self.softmax(fc)
        return y_prob

def train_discriminator(train_x=None, train_y=None, batch_size=1, vocab_size=10):
    if train_x is None:
        x = gen_record(num=batch_size,vocab_size=vocab_size)
    else:
        x = train_x
    if train_y is None:        
       y = gen_label()
    else:
        y = train_y
        
    model = Discriminator(filter_size=FILTER_SIZE, num_filter=NUM_FILTER, vocab_size=vocab_size)
    model = nn.DataParallel(model)
    model.to(DEVICE)
    params = list(filter(lambda p: p.requires_grad, model.parameters()))    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params, lr=0.01)
    
    log = openLog()
    log.write('    training discriminator: {}\n'.format(datetime.now()))
    for epoch in range(DIS_NUM_EPOCH_PRETRAIN):
        pointer = 0
        epoch_loss = []
        while pointer+batch_size <= len(x):
            x_batch = x[pointer:pointer+batch_size]
            y_batch = y[pointer:pointer+batch_size]
            # y_pred dim: (batch_size, nr.of.class)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pointer = pointer + batch_size
            epoch_loss.append(loss.item())
        log.write('      epoch: '+str(epoch)+' loss: '+str(sum(epoch_loss)/len(epoch_loss))+'\n')
    log.close()
    return model

def sanityCheck_discriminator(batch_size=1,vocab_size=10):
    ''' test discriminator instantiation and pretraining'''
    log = openLog('test.txt')
    log.write('\n\nTest discriminator.sanityCheck_discriminator: {}\n'.format(datetime.now()))     
    model = train_discriminator(vocab_size=vocab_size)
    with torch.no_grad():
        x = gen_record(num=batch_size,vocab_size=vocab_size)
        y_pred = model(x)
    log.write('  y_pred shape: '+str(y_pred.shape)+'\n')
    log.close()
    return model, y_pred

#%%
if __name__ == '__main__':
    model, y_pred = sanityCheck_discriminator(batch_size=4,vocab_size=10)
