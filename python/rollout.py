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
    def __init__(self, generator=None, r_update_rate=0.8, vocab_size=10):
        super().__init__()
        if generator is not None:
            self.ignored_tokens = generator.ignored_tokens
            self.embedding = generator.pretrain_model.module.embedding
            self.lstm = copy.deepcopy(generator.pretrain_model.module.lstm)
            self.hidden2tag = copy.deepcopy(generator.pretrain_model.module.hidden2tag)
            self.init_hidden = generator.pretrain_model.module.init_hidden
            self.logSoftmax = generator.pretrain_model.module.logSoftmax
            self.ignoreTokens = generator.ignoreTokens
            self.vocab_size = generator.pretrain_model.module.vocab_size
        else:
            lstm = LSTMCore(vocab_size=vocab_size)
            self.ignored_tokens = None
            self.embedding = lstm.embedding
            self.lstm = lstm.lstm
            self.hidden2tag = lstm.hidden2tag
            self.init_hidden = lstm.init_hidden
            self.logSoftmax = lstm.logSoftmax
            self.vocab_size = vocab_size
            self.ignoreTokens = lambda x, y: x

        self.lstmCell = nn.LSTMCell(EMB_SIZE, GEN_HIDDEN_DIM)
        self.softmax = nn.Softmax(dim=-1)
        self.r_update_rate = r_update_rate

    def forward(self, sentence, hidden, given_num, ignored_tokens=None):
        assert given_num < sentence.shape[-1]
        if ignored_tokens is None:
            ignored_tokens = self.ignored_tokens
        if len(sentence.shape) == 1:
            sentence = sentence.view(1, sentence.shape[0])
        # known input part (the first given_num number of tokens)
        existing = sentence[:,0:given_num]
        embeds_existing = self.embedding(existing.long())
        hidden0 = [x.permute(1,0,2).contiguous() for x in hidden]
        self.lstm_out, hidden0 = self.lstm(embeds_existing, hidden0)
        self.tag_space = self.hidden2tag(self.lstm_out)

        self.y_prob = self.softmax(self.ignoreTokens(self.tag_space, ignored_tokens))
        shape = (self.y_prob.shape[0], self.y_prob.shape[1])
        y_prob_existing_output = self.y_prob.view(-1,self.y_prob.shape[-1]).multinomial(num_samples=1).view(shape)
        y_pred_output = sentence[:,0:given_num]

        # unknown part to be rolled out, in order to estimate the total reward
        x_t = y_prob_existing_output[:,-1].view(-1,1)
        # unpack hidden and cell states from the previous nn.LSTM and reshape for nn.LSTMCell:
        hidden_state, cell_state = hidden0
        hidden_state = hidden_state.view(-1,GEN_HIDDEN_DIM)
        cell_state = cell_state.view(-1,GEN_HIDDEN_DIM)
        hidden0 = (hidden_state, cell_state)
        for i in range(given_num, SEQ_LENGTH):
            embeds_rollout = self.embedding(x_t.long()).view(-1,EMB_SIZE)
            hidden0 = self.lstmCell(embeds_rollout, hidden0)
            tag_space_rollout = self.hidden2tag(hidden0[0])
            y_prob_rollout = self.softmax(self.ignoreTokens(tag_space_rollout,ignored_tokens))
            x_t = y_prob_rollout.multinomial(num_samples=1)
            self.lstm_out = torch.cat((self.lstm_out, hidden0[0].view(-1,1,hidden0[0].shape[-1])),dim=1)
            self.tag_space = torch.cat((self.tag_space, tag_space_rollout.view(-1,1,tag_space_rollout.shape[-1])), dim=1)
            self.y_prob = torch.cat((self.y_prob, y_prob_rollout.view(-1,1,y_prob_rollout.shape[-1])),dim=1)
            y_pred_output = torch.cat((y_pred_output.int(), x_t.int()),dim=1)
        
        tag_scores = self.logSoftmax(self.tag_space)
        return tag_scores, y_pred_output

    def update_params(self, generator):
        for p, w in zip(self.lstm.parameters(), generator.pretrain_model.module.lstm.parameters()):
            p.data = self.r_update_rate * p.data + (1-self.r_update_rate) * w.data
        for p, w in zip(self.hidden2tag.parameters(), generator.pretrain_model.module.hidden2tag.parameters()):
            p.data = self.r_update_rate * p.data + (1-self.r_update_rate) * w.data

def getReward(gen_output, rollout, discriminator):
    with torch.no_grad():
        gen_output = gen_output.view(-1,SEQ_LENGTH)
        batch_size = len(gen_output)
        rewards = torch.zeros(batch_size,SEQ_LENGTH,device=DEVICE)
        # "ROLLOUT_ITER" determines the number of repeated runs of rollout network
        for i in range(ROLLOUT_ITER):
            # The first given_num words are same as input_x,
            #   the last (sequence_length-given_num) words are generated by the lstm network.
            for given_num in range(1, SEQ_LENGTH):
                hidden = rollout.module.init_hidden(batch_size)
                tag_scores, rollout_output = rollout(sentence=gen_output, hidden=hidden, given_num=given_num)
                dis_output = discriminator(rollout_output)
                ypred = [item[1] for item in dis_output]
                # every given_num updates one column, every i in ROLLOUT_ITER updates the entire table.
                rewards[:,given_num-1] += torch.tensor(ypred, device=DEVICE)
            # the last-token-reward
            dis_output = discriminator(gen_output)
            ypred = [item[1] for item in dis_output]
            rewards[:,SEQ_LENGTH-1] += torch.tensor(ypred, device=DEVICE)
        rewards = rewards / (1.0 * ROLLOUT_ITER)
    return rewards

def sanityCheck_rollout(batch_size=5):
    ''' test Rollout instantiation '''
    log = openLog('test.txt')
    log.write('\n\nTest rollout.sanityCheck_rollout: {}'.format(datetime.now()))
    x, _, reverse_vocab, _ = read_sampleFile()
    x0 = x[0:batch_size]
    try:
        model = Rollout(vocab_size=len(reverse_vocab))
        model = nn.DataParallel(model)
        model.to(DEVICE)
        hidden = model.module.init_hidden(len(x0))
        model(x0, hidden, given_num=3)
        log.write('\n  rollout.sanityCheck_rollout SUCCESSFUL: {}\n'.format(datetime.now()))
        log.close()
        return model
    except:
        log.write('\n  rollout.sanityCheck_rollout !!!!!! UNSUCCESSFUL !!!!!! : {}\n'.format(datetime.now()))
        log.close()
        return None

def sanityCheck_rewards(batch_size=5):
    ''' test rewards generation '''
    log = openLog('test.txt')
    log.write('\n\nTest rollout.sanityCheck_rewards: {}'.format(datetime.now()))
    try:
        generator, _, y_output_all = sanityCheck_generator(batch_size=batch_size,sample_size=batch_size*2)
        gen_output = y_output_all[-batch_size:,:]
        rollout = Rollout(generator=generator)
        rollout = nn.DataParallel(rollout)
        rollout.to(DEVICE)
        discriminator = train_discriminator(batch_size=batch_size,vocab_size=generator.pretrain_model.module.vocab_size)
        rewards = getReward(gen_output, rollout, discriminator)
        log.write('\n  rollout.sanityCheck_rewards SUCCESSFUL. {}\n'.format(datetime.now()))
        log.close()
        return rewards
    except:
        log.write('\n  rollout.sanityCheck_rewards !!!!!! UNSUCCESSFUL !!!!!! {}\n'.format(datetime.now()))
        log.close()
        return None

def sanityCheck_rollout_updateParams(batch_size=1):
    ''' test updateParams function '''
    generator, _, _ = sanityCheck_generator(batch_size=batch_size,sample_size=batch_size*2)
    rollout = Rollout(generator=generator)
    rollout = nn.DataParallel(rollout)
    rollout.to(DEVICE)
    log = openLog('test.txt')
    log.write('\n\nTest rollout.sanityCheck_updateParams: {}\n'.format(datetime.now()))
    try:
        log.write('original rollout params:\n')
        param_r = [str(x) for x in list(rollout.module.lstm.parameters())[0][0].tolist()[0:3]]
        log.write(' '.join(param_r))

        generator, _, _ = sanityCheck_generator(model=generator)
        log.write('\nnew generator params:\n')
        param_g = [str(x) for x in list(generator.pretrain_model.module.lstm.parameters())[0][0].tolist()[0:3]]
        log.write(' '.join(param_g))

        rollout.module.update_params(generator)
        log.write('\nnew rollout params:\n')
        param_r = [str(x) for x in list(rollout.module.lstm.parameters())[0][0].tolist()[0:3]]
        log.write(' '.join(param_r))
        log.write('\n  rollout.sanityCheck_updateParams SUCCESSFUL. {}\n'.format(datetime.now()))
    except:
        log.write('\n  rollout.sanityCheck_updateParams !!!!!! UNSUCCESSFUL !!!!!! {}\n'.format(datetime.now()))
    log.close()

#%%
if __name__ == '__main__':
    rollout = sanityCheck_rollout(batch_size=5)
    rewards = sanityCheck_rewards(batch_size=5)
    sanityCheck_rollout_updateParams(batch_size=5)
