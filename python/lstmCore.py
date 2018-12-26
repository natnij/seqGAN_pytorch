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
from config import (SEQ_LENGTH,EMB_SIZE,DEVICE,
                    GEN_HIDDEN_DIM,GEN_NUM_EPOCH_PRETRAIN,openLog)
from data_processing import gen_record,read_sampleFile,decode

def init_matrix(shape, stdDev=0.1):
    normalDistr = Normal(torch.tensor([0.0],device=DEVICE), torch.tensor([stdDev],device=DEVICE))
    normalSample = normalDistr.sample(torch.Size(shape)).squeeze(dim=2)
    return normalSample

class LSTMCore(nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, EMB_SIZE)
        self.lstm = nn.LSTM(EMB_SIZE, GEN_HIDDEN_DIM, batch_first=True)
        self.hidden2tag = nn.Linear(GEN_HIDDEN_DIM, vocab_size)
        self.logSoftmax = nn.LogSoftmax(dim=2)

    def init_hidden(self, batch_size=1):
        # batch_first for slicing between multiple GPUS;
        # to feed into lstm, the hidden dims need to be permutated.
        return (torch.empty(batch_size, 1, 48, device=DEVICE).normal_(),
                torch.empty(batch_size, 1, 48, device=DEVICE).normal_())

    def forward(self, sentence, hidden, sentence_lengths=None):
        # sentence dim: (batch_size, maximum sentence length)        
        if len(sentence.shape) == 1:
            sentence = sentence.view(1,sentence.shape[0])
        if sentence_lengths is None:
            sentence_lengths = torch.LongTensor([sentence.shape[1]] * len(sentence))
        # pack_padded_sequence is not compatible with DataParallel.
        # it needs to be a cpu tensor, and in runtime the model's forward
        # does not slice the cpu tensor as it does the layer's input tensors.
        # work-around as of pytorch 0.4.1: transform the lengths to a cuda tensor 
        # BEFORE entering forward pass, and revert it to a cpu tensor or a list
        # before calling pack_padded_sequence.
        sentence_lengths = sentence_lengths.type(torch.LongTensor)
        if len(sentence_lengths) < len(sentence):
            sentence_lengths = torch.cat([sentence_lengths, torch.LongTensor([sentence.shape[1]]
                                    * (len(sentence)-len(sentence_lengths)))])     
        embeds = self.embedding(sentence.long())
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, sentence_lengths.to(torch.device('cpu')), batch_first=True)
        hidden0 = [x.permute(1,0,2).contiguous() for x in hidden]
        lstm_out, hidden0 = self.lstm(embeds, hidden0)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=sentence.shape[1])
        tag_space = self.hidden2tag(lstm_out)        
        tag_scores = self.logSoftmax(tag_space)
        return tag_scores, tag_space

def pretrain_LSTMCore(train_x=None, sentence_lengths=None, batch_size=1, end_token=None, vocab_size=10):
    if train_x is None:
        x = gen_record(vocab_size=vocab_size)
    else:
        x = train_x
    if len(x.shape) == 1:
        x = x.view(1,x.shape[0])
    if sentence_lengths is None:
        sentence_lengths = [x.shape[1]] * len(x)
    if len(sentence_lengths) < len(x):
        sentence_lengths.extend([x.shape[1]] * (len(x)-len(sentence_lengths)))
    if end_token is None:
        end_token = vocab_size - 1
    
    model = LSTMCore(vocab_size)
    model = nn.DataParallel(model)#, device_ids=[0])
    model.to(DEVICE)
    params = list(filter(lambda p: p.requires_grad, model.parameters()))       
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(params, lr=0.01)
    y_pred_all = []
    log = openLog()
    log.write('    training lstmCore: {}\n'.format(datetime.now()))
    for epoch in range(GEN_NUM_EPOCH_PRETRAIN):
        pointer = 0
        y_pred_all = []
        epoch_loss = []
        while pointer + batch_size <= len(x):
            x_batch = x[pointer:pointer+batch_size]
            x0_length = torch.tensor(sentence_lengths[pointer:pointer+batch_size]).to(device=DEVICE)
            y = torch.cat((x_batch[:,1:],
                           torch.tensor([end_token]*x_batch.shape[0],device=DEVICE)
                           .int().view(x_batch.shape[0],1)),dim=1)
            # hidden has to be passed to the model as a GPU tensor to be correctly sliced between multiple GPUs. 
            # default dim for DataParallel is dim=0, so the inputs will all be sliced on dim0. 
            # so the hidden tensors need to be permutated back to batch-size-second inside the forward pass
            #   in order to feed into the lstm layer. 
            # when using DataParallel the attributes can be accessed through .module
            hidden = model.module.init_hidden(batch_size)            
            y_pred, tag_space = model(x_batch, hidden, x0_length)
            loss = criterion(y_pred.view(-1,y_pred.shape[-1]), y.long().view(-1))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            y_prob = F.softmax(tag_space, dim=2)
            y_pred_all.append(y_prob)
            epoch_loss.append(loss.item())
            pointer = pointer + batch_size
        log.write('      epoch: '+str(epoch)+' loss: '+str(sum(epoch_loss)/len(epoch_loss))+'\n')
    log.close()
    return model, torch.cat(y_pred_all)

def test_genMaxSample(model, start_token=0, batch_size=1):
    ''' test lstmCore's generation function '''
    log = openLog('test.txt')
    log.write('\n\nTest lstmCore.test_genMaxSample: {}'.format(datetime.now()))
    with torch.no_grad():
        y = [start_token] * batch_size
        y_all_max = torch.tensor(y,device=DEVICE).int().view(-1,1)
        hidden = model.module.init_hidden(len(y))
        for i in range(SEQ_LENGTH-1):
            x = torch.tensor(y,device=DEVICE).view([-1,1])
            y_pred, _ = model(x, hidden, sentence_lengths=torch.tensor([1],device=DEVICE).long())
            y_pred = y_pred[:,:,1:-1]
            y_pred = y_pred.squeeze(dim=1)
            # take the max
            y = torch.argmax(y_pred,dim=1).float().view(-1,1)
            y_all_max = torch.cat([y_all_max,y.int()],dim=1)
        
        y = [start_token] * batch_size
        y_all_sample = torch.tensor(y,device=DEVICE).int().view(-1,1)
        hidden = model.module.init_hidden(len(y))
        for i in range(SEQ_LENGTH-1):
            x = torch.tensor(y,device=DEVICE).view([-1,1])
            y_pred, tag_space = model(x,hidden,sentence_lengths=torch.tensor([1],device=DEVICE).long())
            # random choice based on probability distribution.
            y_prob = F.softmax(tag_space, dim=2)
            shape = (y_prob.shape[0],y_prob.shape[1])
            y = y_prob.view(-1,y_prob.shape[-1]).multinomial(num_samples=1).float().view(shape)
            y_all_sample = torch.cat([y_all_sample,y.int()],dim=1)
    log.write('\n  lstmCore.test_genMaxSample SUCCESSFUL: {}\n'.format(datetime.now()))
    log.close()
    return y_all_max, y_all_sample

def sanityCheck_LSTMCore(batch_size=1):
    ''' test prtrain_LSTMCore function '''
    log = openLog('test.txt')
    log.write('\n\nTest lstmCore.sanityCheck_LSTMCore: {}\n'.format(datetime.now())) 
    log.close()
    x, _, reverse_vocab, _ = read_sampleFile()
    pretrain_result = pretrain_LSTMCore(train_x=x,batch_size=batch_size,vocab_size=len(reverse_vocab))
    model = pretrain_result[0]
    y_all_max, y_all_sample = test_genMaxSample(model,start_token=0,batch_size=batch_size)
    log = openLog('test.txt')
    gen_tokens_max = decode(y_all_max, reverse_vocab, log)
    gen_tokens_sample = decode(y_all_sample, reverse_vocab, log)
    log.close()
    return gen_tokens_max, gen_tokens_sample

#%%
if __name__ == '__main__':
    gen_tokens_max, gen_tokens_sample = sanityCheck_LSTMCore(4)
    

