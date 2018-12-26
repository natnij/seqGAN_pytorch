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
from config import (SEQ_LENGTH,DEVICE,GEN_NUM_EPOCH,MAXINT,openLog)
from data_processing import read_sampleFile
from lstmCore import pretrain_LSTMCore

class Generator(nn.Module):
    def __init__(self, pretrain_model=None, start_token=0, 
                 ignored_tokens=None):
        super().__init__()
        self.start_token = start_token
        self.ignored_tokens = ignored_tokens
        if pretrain_model is None:
            x, _, reverse_vocab, _ = read_sampleFile()
            self.pretrain_model, _ = pretrain_LSTMCore(train_x=x, vocab_size=len(reverse_vocab))
        else:
            self.pretrain_model = pretrain_model       
        self.softmax = nn.Softmax(dim=2)
        self.loss = GeneratorLoss()

    def forward(self, x, hidden, rewards, ignored_tokens=None, sentence_lengths=None):
        ''' forward pass. variables can be backpropagated. '''
        if ignored_tokens is None:
            ignored_tokens = self.ignored_tokens
        y, tag_space = self.pretrain_model(x, hidden, sentence_lengths=sentence_lengths)
        y = y.data
        y_pred = self.ignoreTokens(tag_space, ignored_tokens)
        y_prob = self.softmax(y_pred)
        shape = (y_prob.shape[0], y_prob.shape[1])
        try:
            y_output = y_prob.view(-1,y_prob.shape[-1]).multinomial(num_samples=1).view(shape)
        except RuntimeError:
            print('error with multinomial. using argmax instead.')
            y_output = torch.argmax(y_prob.view(-1,y_prob.shape[-1]), dim=1).view(shape)
        if rewards is None:
            rewards = y_prob.sum(dim=2).data
        loss_variable = self.loss(y_prob, x, rewards)
        return y_output, y_prob, loss_variable

    def generate(self, start_token=None, ignored_tokens=None, batch_size=1):
        ''' the generate_LSTMCore only generates samples under torch.no_grad,
            therefore it will not be backpropagated.
        '''
        if start_token is None:
            start_token = self.start_token
        if ignored_tokens is None:
            ignored_tokens = self.ignored_tokens
        y_all_sample = self.generate_LSTMCore(start_token, ignored_tokens, batch_size)
        return y_all_sample
    
    def generate_LSTMCore(self, start_token, ignored_tokens, batch_size=1):
        y = [start_token] * batch_size
        y_all_sample = torch.tensor(y,device=DEVICE).int().view(-1,1)
        with torch.no_grad():
            hidden = self.pretrain_model.module.init_hidden(len(y))
            for i in range(SEQ_LENGTH-1):
                x = torch.tensor(y,device=DEVICE).view([-1,1])
                y_pred, tag_space = self.pretrain_model(x,hidden,sentence_lengths=torch.tensor([1],device=DEVICE).long())
                # random choice based on probability distribution. another possibility would be to take the max.
                y_prob = F.softmax(self.ignoreTokens(tag_space, ignored_tokens), dim=2)
                shape = (y_prob.shape[0], y_prob.shape[1])
                y = y_prob.view(-1,y_prob.shape[-1]).multinomial(num_samples=1).float().view(shape)
                y_all_sample = torch.cat([y_all_sample,y.int()],dim=1)
        return y_all_sample
    
    def ignoreTokens(self, original, ignored_tokens):
        ''' avoid probability of choosing the 'START' or 'END' tokens.
            only call this function in generator and rollout modules. 
            in pretrain_LSTMCore the step is skipped. 
        '''
        if ignored_tokens is None:
            return original
        for token in ignored_tokens:
            if len(original.shape)==3:
                original[:,:,token] = -MAXINT
            else:
                original[:,token] = -MAXINT
        return original

class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

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
        x1 = x.view(-1,1).long()
        pred1 = prediction.view(-1,prediction.shape[-1])
        x2 = self.createOneHotDummy(dim=(x1.shape[0],pred1.shape[1])).scatter_(1,x1,1)

        pred2 = torch.log(torch.clamp(pred1, min=1e-20, max=1.0))
        prod = torch.mul(x2,pred2)
        reduced_prod = torch.sum(prod, dim=1)
        rewards_prod = torch.mul(reduced_prod, rewards.view(-1))
        generator_loss = torch.sum(rewards_prod)
        return generator_loss
    
    def createOneHotDummy(self, dim):
        one_hot = torch.empty(dim, device=DEVICE)
        return one_hot.zero_()

def train_generator(model, x, reward, iter_n_gen=None, batch_size=1, sentence_lengths=None):
    if len(x.shape) == 1:
        x = x.view(1,x.shape[0])
    rem = len(x) % batch_size
    if rem > 0:
        x = x[0:len(x)-rem]
    if sentence_lengths is None:
        sentence_lengths = [x.shape[1]] * len(x)
    if len(sentence_lengths) < len(x):
        sentence_lengths.extend([x.shape[1]] 
                                * (len(x)-len(sentence_lengths)))
    sentence_lengths = torch.tensor(sentence_lengths,device=DEVICE).long()
    if reward is None:
        reward = torch.tensor([1.0] * x.shape[0] * x.shape[1],device=DEVICE).view(x.shape)
    if iter_n_gen is None:
        iter_n_gen = GEN_NUM_EPOCH
        
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.SGD(params, lr=0.01)
    log = openLog()
    log.write('    training generator: {}\n'.format(datetime.now()))
    for epoch in range(iter_n_gen):
        pointer = 0
        y_prob_all = []
        y_output_all = []
        epoch_loss = []
        while pointer + batch_size <= len(x):
            x_batch = x[pointer:pointer+batch_size]
            r_batch = reward[pointer:pointer+batch_size]
            s_length = sentence_lengths[pointer:pointer+batch_size]
            hidden = model.pretrain_model.module.init_hidden(batch_size)
            y_output, y_prob, loss_var = model(x=x_batch, hidden=hidden, rewards=r_batch, sentence_lengths=s_length)
            optimizer.zero_grad()
            loss_var.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            y_prob_all.append(y_prob)
            y_output_all.append(y_output)  
            epoch_loss.append(loss_var.item())
            pointer = pointer + batch_size
        log.write('      epoch: '+str(epoch)+' loss: '+str(sum(epoch_loss)/len(epoch_loss))+'\n')
    log.close()
    return ( model, torch.cat(y_prob_all), torch.cat(y_output_all).view(list(x.shape)) )


def sanityCheck_GeneratorLoss(pretrain_result=None, batch_size=5):
    '''test custom loss function '''
    if pretrain_result is None:
        x, _, reverse_vocab, _ = read_sampleFile()
        pretrain_result = pretrain_LSTMCore(x,vocab_size=len(reverse_vocab))
    model = pretrain_result[0]
    y_pred_pretrain = pretrain_result[1].view([-1,SEQ_LENGTH,len(reverse_vocab)])
    test_reward = y_pred_pretrain.sum(dim=2).data
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.SGD(params, lr=0.01)
    optimizer.zero_grad()
    
    log = openLog('test.txt')
    log.write('\n\nTest generator.sanityCheck_GeneratorLoss: {}\n'.format(datetime.now())) 
    criterion = GeneratorLoss()
    g_loss = criterion(y_pred_pretrain[0:batch_size,:,:], 
                      x[0:batch_size,:], test_reward[0:batch_size,:])  
    g_loss.backward()
    optimizer.step()
    log.write('  generator.sanityCheck_GeneratorLoss SUCCESSFUL: '+str(g_loss)+'\n')
    log.close()
    return g_loss

def sanityCheck_generator(model=None, batch_size=1, sample_size=5):
    ''' test Generator instantiation and train_generator function '''
    log = openLog('test.txt')
    log.write('\n\nTest generator.sanityCheck_generator: {}\n'.format(datetime.now()))
    x, vocabulary, reverse_vocab, _ = read_sampleFile(num=sample_size)
    if model is None:
        pretrain_result = pretrain_LSTMCore(x,vocab_size=len(vocabulary))
        model = Generator(pretrain_model=pretrain_result[0])
        log.write('  generator instantiated: {}\n'.format(datetime.now()))
        model.to(DEVICE)
    model, y_prob_all, y_output_all = train_generator(model, x, reward=None, batch_size=batch_size)
    log.write('  trained generator outputs:\n')
    log.write('    y_output_all shape: '+ str(y_output_all.shape) +'\n')
    log.write('    y_prob_all shape: '+ str(y_prob_all.shape) +'\n')
    log.close()
    return model, y_prob_all, y_output_all

#%%
if __name__ == '__main__':
    model, y_prob_all, y_output_all = sanityCheck_generator(batch_size=5, sample_size=12)
        