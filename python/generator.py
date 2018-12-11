# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 13:28:02 2018
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
from config import (SEQ_LENGTH,BATCH_SIZE,VOCAB_SIZE, DEVICE,GEN_NUM_EPOCH, 
                    ONEHOT_SINGLE,MAXINT,openLog)
from data_processing import read_sampleFile
from lstmCore import pretrain_LSTMCore

class Generator(nn.Module):
    def __init__(self, pretrain_model=None, start_token=0, ignored_tokens=None):
        super(Generator, self).__init__()
        self.start_token = start_token
        self.ignored_tokens = ignored_tokens
        if pretrain_model is None:
            x, _, reverse_vocab = read_sampleFile()
            self.pretrain_model, _ = pretrain_LSTMCore(x)
        else:
            self.pretrain_model = pretrain_model       
        self.softmax = nn.Softmax(dim=1)
        self.loss = GeneratorLoss()
    
    def forward(self, x, rewards, ignored_tokens=None):
        ''' forward pass. variables can be backpropagated.
        '''
        if ignored_tokens is None:
            ignored_tokens = self.ignored_tokens
        y = self.pretrain_model(x).data
        y_pred = self.pretrain_model.tag_space
        y_pred = self.ignoreTokens(y_pred, ignored_tokens)
        self.y_prob = self.softmax(y_pred)
        self.y_output = self.y_prob.multinomial(num_samples=1)
        
        if rewards is None:
            rewards = self.y_prob.sum(dim=1).data
            
        self.loss_variable = self.loss(self.y_prob, x, rewards)     
        return self.y_output
    
    def generate(self, start_token=None, ignored_tokens=None):
        ''' the generate_LSTMCore only generates samples under torch.no_grad,
            therefore it will not be backpropagated.
        '''
        if start_token is None:
            start_token = self.start_token
        if ignored_tokens is None:
            ignored_tokens = self.ignored_tokens
        y_all_sample = self.generate_LSTMCore(start_token, ignored_tokens)
        return y_all_sample
    
    def generate_LSTMCore(self, start_token, ignored_tokens):
        y = start_token
        y_all_sample = [int(start_token)]
        with torch.no_grad():            
            self.pretrain_model.hidden = self.pretrain_model.init_hidden()
            for i in range(SEQ_LENGTH-1):        
                x = torch.Tensor([y]).view([-1,1])
                y_pred = self.pretrain_model(x)
                # random choice based on probability distribution. another possibility would be to take the max.
                y_prob = F.softmax(self.ignoreTokens(self.pretrain_model.tag_space, ignored_tokens), dim=1)
                y_prob = y_prob.squeeze(dim=0)
                y = y_prob.multinomial(num_samples=1)
                y_all_sample.append(int(y.tolist()[0]))
        return y_all_sample
    
    def ignoreTokens(self, original, ignored_tokens):
        ''' avoid probability of choosing the 'START' or 'END' tokens.
            only call this function in generator and rollout modules. 
            in pretrain_LSTMCore the step is skipped. 
        '''
        if ignored_tokens is None:
            return original
        A = torch.eye(n=original.shape[1])
        for token in ignored_tokens:
            if A[token][token] < 0:
                A[token][token] = MAXINT
            else:
                A[token][token] = -MAXINT
        return torch.mm(original, A)
        

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss,self).__init__()

    def forward(self, prediction, x, rewards):
        '''
        dimension calculation:
         x: dim(batch, seq_length)
         x1 = reshape(x): dim(batch * seq_length), 1-dimensional vector
         x2 = one_hot(x1): dim(batch * seq_length, vocab_size), 2-dimensional
         g_predictions: dim(batch, seq_length, vocab_size), 3-dimensional
         pred1 = reshape(g_predictions): dim(batch * seq_length, vocab_size), 
             2-dimensional
         pred2 = log(clip_by_value(pred1)): dim(batch * seq_length, vocab_size)
         reduced_pred = reduce_sum(x * pred2, axis=1): dim(batch * seq_length), 
             1-dim vector, summed along axis=1
         rewards: dim(batch, seq_length)
         reshaped_rewards = reshape(rewards): dim(batch * seq_length)
         g_loss = -reduce_sum(reduced_pred * reshaped_rewards): 
             sumproduct of the two 1-dimensional vectors. 
             g_loss reduces to one single value.
        '''
        x1 = x.view([-1,1]).long()
        ONEHOT_SINGLE.zero_()
        x2 = ONEHOT_SINGLE.scatter_(1,x1,1)

        pred1 = prediction.view([-1,VOCAB_SIZE])
        # equivalent to tensorflow.clip_by_value
        pred2 = torch.log(torch.clamp(pred1, min=1e-20, max=1.0))
        prod = torch.mul(x2,pred2)
        reduced_prod = torch.sum(prod, dim=1)
        rewards_prod = torch.mul(reduced_prod, rewards.view([-1]))
        generator_loss = torch.sum(rewards_prod)
        return generator_loss

def train_generator(model, x, reward, iter_n_gen=None):
    if reward is None:
        reward = [None] * len(x)
    if iter_n_gen is None:
        iter_n_gen = GEN_NUM_EPOCH
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.SGD(params, lr=0.01)
    log = openLog()
    log.write('\n\ntraining generator: {}\n'.format(datetime.now()))
    for epoch in range(iter_n_gen):
        y_prob_all = []
        y_output_all = []
        epoch_loss = []
        for i,(x0,r0) in enumerate(zip(x,reward)):
            y_output = model(x0, r0)
            y_prob = model.y_prob
            loss_var = model.loss_variable
            optimizer.zero_grad()
            loss_var.backward()
            optimizer.step()
            model.pretrain_model.hidden = model.pretrain_model.init_hidden()
            y_prob_all.append(y_prob)
            y_output_all.append(y_output)  
            epoch_loss.append(loss_var.item())
        log.write('epoch: '+str(epoch)+' loss: '+str(sum(epoch_loss)/len(epoch_loss))+'\n')
    log.close()
    return ( model, torch.cat(y_prob_all), torch.cat(y_output_all).view(list(x.shape)) )


def sanityCheck_GeneratorLoss(pretrain_result=None):
    '''test custom loss function '''
    if pretrain_result is None:
        x, _, reverse_vocab = read_sampleFile()
        pretrain_result = pretrain_LSTMCore(x)
    model = pretrain_result[0]
    y_pred_pretrain = pretrain_result[1].view([-1,SEQ_LENGTH,VOCAB_SIZE])
    test_reward = y_pred_pretrain.sum(dim=2).data
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.SGD(params, lr=0.01)
    optimizer.zero_grad()
    
    log = openLog('test.txt')
    log.write('\n\nTest generator.sanityCheck_GeneratorLoss: {}\n'.format(datetime.now())) 
    criterion = GeneratorLoss()
    g_loss = criterion(y_pred_pretrain[0:BATCH_SIZE,:,:], 
                      x[0:BATCH_SIZE,:], test_reward[0:BATCH_SIZE,:])  
    g_loss.backward()
    optimizer.step()
    log.write('  generator.sanityCheck_GeneratorLoss SUCCESSFUL: '+str(g_loss)+'\n')
    log.close()
    return g_loss

def sanityCheck_generator(model=None):
    ''' test Generator instantiation and train_generator function '''
    log = openLog('test.txt')
    log.write('\n\nTest generator.sanityCheck_generator: {}\n'.format(datetime.now()))     
    x, vocabulary, reverse_vocab = read_sampleFile()
    if model is None:
        pretrain_result = pretrain_LSTMCore(x)
        model = Generator(pretrain_model=pretrain_result[0])
        log.write('  generator instantiated: {}\n'.format(datetime.now()))  
    model.to(DEVICE)
    model, y_prob_all, y_output_all = train_generator(model, x, reward=None)
    log.write('  trained generator outputs:\n')
    log.write('    y_output_all shape: '+ str(y_output_all.shape) +'\n')
    log.write('    y_prob_all shape: '+ str(y_prob_all.shape) +'\n')
    log.close()
    return model, y_prob_all, y_output_all

#%%
if __name__ == '__main__':
    model, y_prob_all, y_output_all = sanityCheck_generator()
        