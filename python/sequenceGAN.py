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
import torch
from config import BATCH_SIZE, TOTAL_BATCH, DIS_NUM_EPOCH, DEVICE, openLog
from data_processing import gen_label
from lstmCore import read_sampleFile, pretrain_LSTMCore
from discriminator import train_discriminator
from generator import Generator, train_generator
from rollout import Rollout, getReward

def generate_samples(genModel, gen_num):
    x_gen = []
    for i in range(gen_num):
        x_gen.append(genModel.generate())
        x_gen = [torch.Tensor(t) for t in x_gen]
    return torch.t(torch.stack(x_gen,dim=1))

def pretrain_generator(x, start_token=0, ignored_tokens=None):
    pretrain_result = pretrain_LSTMCore(x)
    generator = Generator(pretrain_model=pretrain_result[0], start_token=start_token, ignored_tokens=ignored_tokens)
    generator.to(DEVICE)
    return generator

def pretrain_discriminator(x, x_gen):
    y = gen_label(len(x),fixed_value=1)
    y_gen = gen_label(len(x_gen),fixed_value=0)
    x_train = torch.cat([x.int(),x_gen.int()], dim=0)
    y_train = torch.cat([y,y_gen], dim=0)
    discriminator = train_discriminator(x_train, y_train)
    return discriminator

def main():
    x, vocabulary, reverse_vocab = read_sampleFile()
    ignored_tokens = [vocabulary['START'], vocabulary['END']]
    start_token = vocabulary['START']
    generator = pretrain_generator(x, start_token=start_token, ignored_tokens=ignored_tokens)
    x_gen = generate_samples(genModel=generator, gen_num=len(x))
    discriminator = pretrain_discriminator(x, x_gen)
    rollout = Rollout(generator, 0.8)
    rollout.to(DEVICE)
    for total_batch in range(TOTAL_BATCH):
        print('batch: {}'.format(total_batch))
        for it in range(1):
            # generate the entire sequence
            samples = generate_samples(genModel=generator, gen_num=BATCH_SIZE)
            # take the rewards as average of ROLLOUT_ITER times.
            # The more times a [0,1] class (positive, real data) 
            #   is returned, the higher the reward. 
            rewards = []
            for sample in samples:
                rewards.append(getReward(sample, rollout, discriminator))
            # https://stackoverflow.com/questions/48915810/pytorch-contiguous
            rewards = torch.t(torch.stack(rewards,dim=1)).contiguous()
            (generator, 
             y_prob_all, y_output_all) = train_generator(generator, samples, rewards, iter_n_gen=1)
        
        rollout.update_params(generator)
        
        for iter_n_dis in range(DIS_NUM_EPOCH):
            print('iter_n_dis: {}'.format(iter_n_dis))
            x_gen = generate_samples(genModel=generator, gen_num=len(x))
            discriminator = pretrain_discriminator(x, x_gen)
    
    log = openLog('genTxt.txt')
    gen_tokens_final = []
    for i in range(8):
        num = generator.generate()[1:]
        words = []
        for n in num:
            words.append(reverse_vocab[n])
        log.write(' '.join(words)+'\n')
        gen_tokens_final.append(words)
    log.close()
    print(gen_tokens_final)
            
    

#%%
if __name__ == '__main__':
    main()
    
    