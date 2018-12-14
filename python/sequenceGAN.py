# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:16:41 2018
@author: natnij

Based on SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient, 
    Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu.
    Paper available here: https://arxiv.org/abs/1609.05473
Translated from the original tensorflow repo: 
    https://github.com/LantaoYu/SeqGAN, and adjusted for wider usability.
Many thanks to the original authors. 
"""
import sys
import torch
from config import TOTAL_BATCH, DIS_NUM_EPOCH, DEVICE, openLog
from data_processing import gen_label,decode
from lstmCore import read_sampleFile, pretrain_LSTMCore
from discriminator import train_discriminator
from generator import Generator, train_generator
from rollout import Rollout, getReward

def pretrain_generator(x,start_token,end_token,ignored_tokens=None,
                       sentence_lengths=None,batch_size=1):
    pretrain_result = pretrain_LSTMCore(train_x=x,
                            sentence_lengths=sentence_lengths, 
                            batch_size=batch_size, end_token=end_token)
    generator = Generator(pretrain_model=pretrain_result[0], 
                start_token=start_token, ignored_tokens=ignored_tokens)    
    generator.to(DEVICE)
    return generator

def train_discriminator_wrapper(x, x_gen, batch_size=1):
    y = gen_label(len(x),fixed_value=1)
    y_gen = gen_label(len(x_gen),fixed_value=0)
    x_train = torch.cat([x.int(),x_gen.int()], dim=0)
    y_train = torch.cat([y,y_gen], dim=0)
    discriminator = train_discriminator(x_train, y_train, batch_size)
    return discriminator

def main(batch_size):
    if batch_size is None:
        batch_size = 1
    x, vocabulary, reverse_vocab, sentence_lengths = read_sampleFile()
    start_token = vocabulary['START']
    end_token = vocabulary['END']
    ignored_tokens = [start_token, end_token]
    
    generator = pretrain_generator(x, start_token=start_token, 
                    end_token=end_token,ignored_tokens=ignored_tokens,
                    sentence_lengths=sentence_lengths,batch_size=batch_size)
    x_gen = generator.generate(start_token=start_token, ignored_tokens=ignored_tokens, 
                               batch_size=len(x))
    discriminator = train_discriminator_wrapper(x, x_gen, batch_size)
    rollout = Rollout(generator, 0.8)
    rollout.to(DEVICE)
    for total_batch in range(TOTAL_BATCH):
        print('batch: {}'.format(total_batch))
        for it in range(1):
            samples = generator.generate(start_token=start_token, 
                    ignored_tokens=ignored_tokens, batch_size=batch_size)
            # Take average of ROLLOUT_ITER times of rewards.
            #   The more times a [0,1] class (positive, real data) 
            #   is returned, the higher the reward. 
            rewards = getReward(samples, rollout, discriminator)
            (generator, y_prob_all, y_output_all) = train_generator(model=generator, x=samples, 
                    reward=rewards, iter_n_gen=1, batch_size=batch_size, sentence_lengths=sentence_lengths)
        
        rollout.update_params(generator)
        
        for iter_n_dis in range(DIS_NUM_EPOCH):
            print('iter_n_dis: {}'.format(iter_n_dis))
            x_gen = generator.generate(start_token=start_token, ignored_tokens=ignored_tokens, 
                               batch_size=len(x))
            discriminator = train_discriminator_wrapper(x, x_gen, batch_size)
    
    log = openLog('genTxt.txt')
    num = generator.generate(batch_size=batch_size)
    words_all = decode(num, reverse_vocab, log)
    log.close()
    print(words_all)

#%%
if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1
    main(batch_size)
    
    