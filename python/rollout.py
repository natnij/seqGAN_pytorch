# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 23:10:00 2018
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
from config import (SEQ_LENGTH,EMB_SIZE, GEN_HIDDEN_DIM, 
                    ROLLOUT_ITER, DEVICE, openLog)
from lstmCore import LSTMCore, read_sampleFile
from discriminator import train_discriminator
from generator import sanityCheck_generator
import copy

class Rollout(nn.Module):
    def __init__(self, generator=None, r_update_rate=0.8):
        super(Rollout, self).__init__()
        if generator is not None:
            self.ignored_tokens = generator.ignored_tokens
            self.embedding = generator.pretrain_model.embedding
            # deepcopy the layer, otherwise it's only a pointer
            self.lstm = copy.deepcopy(generator.pretrain_model.lstm)
            self.hidden2tag = copy.deepcopy(generator.pretrain_model.hidden2tag)
            self.init_hidden = generator.pretrain_model.init_hidden
            self.logSoftmax = generator.pretrain_model.logSoftmax
            self.ignoreTokens = generator.ignoreTokens
        else:
            lstm = LSTMCore()
            self.ignored_tokens = None
            self.embedding = lstm.embedding
            self.lstm = lstm.lstm
            self.hidden2tag = lstm.hidden2tag
            self.init_hidden = lstm.init_hidden
            self.logSoftmax = lstm.logSoftmax
            self.ignoreTokens = lambda x, y: x # if not defined, then include all tokens
            
        self.lstmCell = nn.LSTMCell(EMB_SIZE, GEN_HIDDEN_DIM)
        self.softmax = nn.Softmax(dim=1)
        self.r_update_rate = r_update_rate
    
    def forward(self, sentence, given_num, ignored_tokens=None):
        assert given_num < len(sentence)
        if ignored_tokens is None:
            ignored_tokens = self.ignored_tokens
        # known input part (the first given_num number of tokens)
        existing = sentence[0:given_num]
        embeds_existing = self.embedding(existing)
        self.lstm_out, self.hidden = self.lstm(embeds_existing.view(len(existing), 1, -1), self.hidden)
        self.lstm_out = self.lstm_out.view(len(existing), -1)
        self.tag_space = self.hidden2tag(self.lstm_out)
        
        self.y_prob = self.softmax(self.ignoreTokens(self.tag_space, ignored_tokens))
        y_prob_existing_output = self.y_prob.multinomial(num_samples=1)
        self.y_pred_output = sentence[0:given_num]
        
        # unknown part to be rolled out, in order to estimate the total reward,
        #   start token for rollout part is the last hidden state from the known input part
        x_t = y_prob_existing_output[-1]
        # unpack hidden and cell states from the previous nn.LSTM and reshape for nn.LSTMCell:
        hidden_state, cell_state = self.hidden
        hidden_state = hidden_state.view(1,-1)
        cell_state = cell_state.view(1,-1)
        self.hidden = (hidden_state, cell_state)
        for i in range(given_num, SEQ_LENGTH):
            embeds_rollout = self.embedding(x_t)
            self.hidden = self.lstmCell(embeds_rollout.view(1,-1), self.hidden)
            tag_space_rollout = self.hidden2tag(self.hidden[0])
            y_prob_rollout = self.softmax(self.ignoreTokens(tag_space_rollout,ignored_tokens))
            x_t = y_prob_rollout.multinomial(num_samples=1)
            self.lstm_out = torch.cat((self.lstm_out, self.hidden[0]),dim=0)
            self.tag_space = torch.cat((self.tag_space, tag_space_rollout), dim=0)
            self.y_prob = torch.cat((self.y_prob, y_prob_rollout),dim=0)
            self.y_pred_output = torch.cat((self.y_pred_output.int(), x_t.int().view(1)))
        
        tag_scores = self.logSoftmax(self.tag_space)
        return tag_scores
    
    def update_params(self, generator):
        for p, w in zip(self.lstm.parameters(), generator.pretrain_model.lstm.parameters()):
            p.data = self.r_update_rate * p.data + (1-self.r_update_rate) * w.data
        for p, w in zip(self.hidden2tag.parameters(), generator.pretrain_model.hidden2tag.parameters()):
            p.data = self.r_update_rate * p.data + (1-self.r_update_rate) * w.data 

def getReward(gen_output, rollout, discriminator):
    with torch.no_grad():
        rewards = []
        gen_output = gen_output.view(-1)
        # "ROLLOUT_ITER" determines the number of repeated runs of rollout network
        #   to account for randomness. 
        for i in range(ROLLOUT_ITER):
            # given_num is between 1 and sequence_length-1 for a partially completed sentence.
            # Every iteration of the inside loop creates a complete
            #   sequence, the first given_num words are same as input_x,
            #   the last words are generated by the lstm network.
            # Then in the same iteration the discriminator will be run with the sequence generated,
            #   and probability for the true class will be saved in "rewards".
            for given_num in range(1, SEQ_LENGTH):
                rollout.hidden = rollout.init_hidden()
                tag_scores = rollout(gen_output, given_num)
                rollout_output = rollout.y_pred_output
                dis_output = discriminator(rollout_output)
                ypred = [item[1] for item in dis_output][0]
                if i==0:
                    rewards.append(ypred)
                else:
                    rewards[given_num-1] += ypred
            # the last-token-reward
            dis_output = discriminator(gen_output)
            ypred = [item[1] for item in dis_output][0]
            if i==0:
                rewards.append(ypred)
            else:
                rewards[SEQ_LENGTH-1] += ypred
        rewards = [item / (1.0 * ROLLOUT_ITER) for item in rewards]
    return torch.stack(rewards)
    
def sanityCheck_rollout():
    ''' test Rollout instantiation '''
    log = openLog('test.txt')
    log.write('\n\nTest rollout.sanityCheck_rollout: {}'.format(datetime.now()))
    x, _, reverse_vocab = read_sampleFile()
    x0 = x[0]
    try:
        model = Rollout()
        model.to(DEVICE)
        model(x0,given_num=3)
        log.write('\n  rollout.sanityCheck_rollout SUCCESSFUL: {}\n'.format(datetime.now()))
        log.close()
        return model
    except:
        log.write('\n  rollout.sanityCheck_rollout UNSUCCESSFUL: {}\n'.format(datetime.now()))
        log.close()
        return None

def sanityCheck_rewards():
    ''' test rewards generation '''
    log = openLog('test.txt')
    log.write('\n\nTest rollout.sanityCheck_rewards: {}'.format(datetime.now()))
    try:
        generator, _, y_output_all = sanityCheck_generator()
        gen_output = y_output_all[-SEQ_LENGTH:]
        rollout = Rollout(generator=generator)
        rollout.to(DEVICE)
        discriminator = train_discriminator()    
        rewards = getReward(gen_output, rollout, discriminator)
        log.write('\n  rollout.sanityCheck_rewards SUCCESSFUL. {}\n'.format(datetime.now()))
        log.close()
        return rewards
    except:
        log.write('\n  rollout.sanityCheck_rewards UNSUCCESSFUL. {}\n'.format(datetime.now()))
        log.close()
        return None

def sanityCheck_rollout_updateParams():
    ''' test updateParams function '''
    generator, _, _ = sanityCheck_generator()
    rollout = Rollout(generator=generator)
    rollout.to(DEVICE)
    log = openLog('test.txt')
    log.write('\n\nTest rollout.sanityCheck_updateParams: {}\n'.format(datetime.now()))
    log.write('original rollout params:\n')
    param_r = [str(x) for x in list(rollout.lstm.parameters())[0][0].tolist()]
    log.write(' '.join(param_r))
    
    generator, _, _ = sanityCheck_generator(model=generator)
    log.write('\nnew generator params:\n')
    param_g = [str(x) for x in list(generator.pretrain_model.lstm.parameters())[0][0].tolist()]
    log.write(' '.join(param_g))
    
    rollout.update_params(generator)
    log.write('\nnew rollout params:\n')
    param_r = [str(x) for x in list(rollout.lstm.parameters())[0][0].tolist()]
    log.write(' '.join(param_r))
    log.close()

#%%
if __name__ == '__main__':
    rollout = sanityCheck_rollout()
    rewards = sanityCheck_rewards()
    sanityCheck_rollout_updateParams()
