# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 08:51:45 2018
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
from torch.distributions import Normal
from config import (SEQ_LENGTH,EMB_SIZE,VOCAB_SIZE,
                    DEVICE,GEN_HIDDEN_DIM,GEN_NUM_EPOCH_PRETRAIN,openLog)
from embedding import Embedding
from data_processing import gen_record,read_sampleFile

def init_matrix(shape, stdDev=0.1):
    normalDistr = Normal(torch.tensor([0.0]), torch.tensor([stdDev]))
    normalSample = normalDistr.sample(torch.Size(shape)).squeeze(dim=2)
    return normalSample

class LSTMCore(nn.Module):
    def __init__(self):
        super(LSTMCore, self).__init__()
        self.embedding = Embedding(VOCAB_SIZE, EMB_SIZE)
        self.lstm = nn.LSTM(EMB_SIZE, GEN_HIDDEN_DIM)
        self.hidden2tag = nn.Linear(GEN_HIDDEN_DIM, VOCAB_SIZE)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(1, 1, GEN_HIDDEN_DIM), torch.randn(1, 1, GEN_HIDDEN_DIM))
        
    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        self.tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = self.logSoftmax(self.tag_space)
        return tag_scores

def pretrain_LSTMCore(train_x=None):
    if train_x is None:
        x = gen_record()
    else:
        x = train_x
    
    model = LSTMCore()
    model.to(DEVICE)
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(params, lr=0.01)
    y_pred_all = []
    log = openLog()
    log.write('\n\ntraining lstmCore: {}\n'.format(datetime.now()))
    for epoch in range(GEN_NUM_EPOCH_PRETRAIN):
        y_pred_all = []
        epoch_loss = []
        for i,x0 in enumerate(x):
            y = torch.cat((x0[1:],torch.Tensor([VOCAB_SIZE-1]).int()),dim=0)
            y_pred = model(x0)
            loss = criterion(y_pred, y.long())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            model.hidden = model.init_hidden()
            y_prob = F.softmax(model.tag_space, dim=1)
            y_pred_all.append(y_prob)
            epoch_loss.append(loss.item())
        log.write('epoch: '+str(epoch)+' loss: '+str(sum(epoch_loss)/len(epoch_loss))+'\n')
    log.close()
    return model, torch.cat(y_pred_all)

def test_genMaxSample(model, start_token=0):
    ''' test lstmCore's generation function '''
    log = openLog('test.txt')
    log.write('\n\nTest lstmCore.test_genMaxSample: {}'.format(datetime.now()))    
    y = start_token
    y_all_max = [int(start_token)]
    y_all_sample = [int(start_token)]
    with torch.no_grad():
        model.hidden = model.init_hidden()
        for i in range(SEQ_LENGTH-1):
            x = torch.Tensor([y]).view([-1,1])
            y_pred = model(x)
            y_pred = y_pred.squeeze(dim=0)
            y_pred = y_pred[0:-1]
            # take the max. another possibility would be random choice based on probability distribution.
            y = torch.argmax(y_pred,dim=0)
            y_all_max.append(int(y))
        
        model.hidden = model.init_hidden()
        for i in range(SEQ_LENGTH-1):        
            x = torch.Tensor([y]).view([-1,1])
            y_pred = model(x)
            # random choice based on probability distribution. another possibility would be to take the max.
            y_prob = F.softmax(model.tag_space, dim=1)
            y_prob = y_prob.squeeze(dim=0)
            y = y_prob.multinomial(num_samples=1)
            y_all_sample.append(int(y.tolist()[0]))
    log.write('\n  lstmCore.test_genMaxSample SUCCESSFUL. {}\n'.format(datetime.now()))
    log.close()
    return y_all_max, y_all_sample

def sanityCheck_LSTMCore():
    ''' test prtrain_LSTMCore function '''
    log = openLog('test.txt')
    log.write('\n\nTest lstmCore.sanityCheck_LSTMCore: {}\n'.format(datetime.now())) 
    x, _, reverse_vocab = read_sampleFile()
    pretrain_result = pretrain_LSTMCore(x)
    model = pretrain_result[0]
    y_all_max, y_all_sample = test_genMaxSample(model)
    gen_tokens_max = [reverse_vocab[w] for w in y_all_max]
    gen_tokens_sample = [reverse_vocab[w] for w in y_all_sample]
    log.write('  gen_tokens_max: ' + '_'.join(gen_tokens_max) + '\n')
    log.write('  gen_tokens_sample: ' + '_'.join(gen_tokens_sample) + '\n')
    log.close()
    return gen_tokens_max, gen_tokens_sample

#%%
if __name__ == '__main__':
    gen_tokens_max, gen_tokens_sample = sanityCheck_LSTMCore()
    

