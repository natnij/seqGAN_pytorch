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
from datetime import datetime
import torch
from config import TOTAL_BATCH, DIS_NUM_EPOCH, DEVICE, PATH, NrGPU, openLog
from data_processing import gen_label,decode
from lstmCore import read_sampleFile, pretrain_LSTMCore
from discriminator import train_discriminator
from generator import Generator, train_generator
from rollout import Rollout, getReward

def pretrain_generator(x,start_token,end_token,ignored_tokens=None,
                       sentence_lengths=None,batch_size=1,vocab_size=10):
    pretrain_result = pretrain_LSTMCore(train_x=x,
                            sentence_lengths=sentence_lengths, 
                            batch_size=batch_size, end_token=end_token,
                            vocab_size=vocab_size)
    generator = Generator(pretrain_model=pretrain_result[0],
                start_token=start_token, ignored_tokens=ignored_tokens)
    # generator is not DataParallel. the lstmCore inside is. 
    # if generator is also DataParallel, when it calls lstmCore it invokes the
    #   error message "RuntimeError: all tensors must be on devices[0]"
    #   because the generator instance may not be on devices[0].
    generator.to(DEVICE)
    return generator

def train_discriminator_wrapper(x, x_gen, batch_size=1, vocab_size=10):
    y = gen_label(len(x),fixed_value=1)
    y_gen = gen_label(len(x_gen),fixed_value=0)
    x_train = torch.cat([x.int(),x_gen.int()], dim=0)
    y_train = torch.cat([y,y_gen], dim=0)
    discriminator = train_discriminator(x_train, y_train, batch_size, vocab_size)
    return discriminator

def main(batch_size, num=None):
    if batch_size is None:
        batch_size = 1
    x, vocabulary, reverse_vocab, sentence_lengths = read_sampleFile(num=num)
    if batch_size > len(x):
        batch_size = len(x)
    start_token = vocabulary['START']
    end_token = vocabulary['END']
    pad_token = vocabulary['PAD']
    ignored_tokens = [start_token, end_token, pad_token]
    vocab_size = len(vocabulary)
    
    log = openLog()
    log.write("###### start to pretrain generator: {}\n".format(datetime.now()))
    log.close()
    generator = pretrain_generator(x, start_token=start_token,
                    end_token=end_token,ignored_tokens=ignored_tokens,
                    sentence_lengths=torch.tensor(sentence_lengths,device=DEVICE).long(),
                    batch_size=batch_size,vocab_size=vocab_size)
    x_gen = generator.generate(start_token=start_token, ignored_tokens=ignored_tokens,
                               batch_size=len(x))
    log = openLog()
    log.write("###### start to pretrain discriminator: {}\n".format(datetime.now()))
    log.close()
    discriminator = train_discriminator_wrapper(x, x_gen, batch_size, vocab_size)
    rollout = Rollout(generator, r_update_rate=0.8)
    rollout = torch.nn.DataParallel(rollout)#, device_ids=[0])
    rollout.to(DEVICE)

    log = openLog()
    log.write("###### start to train adversarial net: {}\n".format(datetime.now()))
    log.close()
    for total_batch in range(TOTAL_BATCH):
        log = openLog()
        log.write('batch: {} : {}\n'.format(total_batch, datetime.now()))
        print('batch: {} : {}\n'.format(total_batch, datetime.now()))
        log.close()
        for it in range(1):
            samples = generator.generate(start_token=start_token,
                    ignored_tokens=ignored_tokens, batch_size=batch_size)
            # Take average of ROLLOUT_ITER times of rewards.
            #   The more times a [0,1] class (positive, real data)
            #   is returned, the higher the reward.
            rewards = getReward(samples, rollout, discriminator)
            (generator, y_prob_all, y_output_all) = train_generator(model=generator, x=samples,
                    reward=rewards, iter_n_gen=1, batch_size=batch_size, sentence_lengths=sentence_lengths)

        rollout.module.update_params(generator)

        for iter_n_dis in range(DIS_NUM_EPOCH):
            log = openLog()
            log.write('  iter_n_dis: {} : {}\n'.format(iter_n_dis, datetime.now()))
            log.close()
            x_gen = generator.generate(start_token=start_token, ignored_tokens=ignored_tokens,
                               batch_size=len(x))
            discriminator = train_discriminator_wrapper(x, x_gen, batch_size,vocab_size)

    log = openLog()
    log.write('###### training done: {}\n'.format(datetime.now()))
    log.close()
    
    torch.save(reverse_vocab, PATH+'reverse_vocab.pkl')
    try:
        torch.save(generator, PATH+'generator.pkl')
        print('successfully saved generator model.')
    except:
        print('error: model saving failed!!!!!!')

    log = openLog('genTxt.txt')
    num = generator.generate(batch_size=batch_size)
    log.close()
#    words_all = decode(num, reverse_vocab, log)
#    print(words_all)

#%%
if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1
    try:
        num = int(sys.argv[2])
    except IndexError:
        num=10
    if batch_size<NrGPU:
        batch_size = NrGPU
        
    main(batch_size,num)


