# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:17:25 2018
@author: natnij

Based on SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient, 
    Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu.
    Paper available here: https://arxiv.org/abs/1609.05473
Translated from the original tensorflow repo: 
    https://github.com/LantaoYu/SeqGAN, and adjusted for wider usability.
Many thanks to the original authors.
"""
import pandas as pd
from itertools import chain
from collections import Counter
import torch
from config import SEQ_LENGTH,GENERATE_NUM,DEVICE,PATH

def gen_record(num=GENERATE_NUM, vocab_size=10):
    # batch x nChannels x Height x Width
    data = torch.rand(num, SEQ_LENGTH-1, device=DEVICE)
    data = torch.abs(data * (vocab_size-2)).int()+1
    data = torch.cat([torch.zeros([num,1],device=DEVICE).int(), data], dim=1)
    return data

def gen_label(num=GENERATE_NUM, target_space=2, fixed_value=None):
    # the loss function's target should be a torch.LongTensor.
    # the target's dimension should be only 1D with class index (target_space).
    if fixed_value is None:
        return torch.randint(low=0, high=target_space, size=(num,), device=DEVICE).long()
    else:
        assert fixed_value < target_space
        return torch.randint(low=fixed_value, high=fixed_value+1, size=(num,), device=DEVICE).long()

def read_sampleFile(file='real_data_london.pkl', pad_token='PAD', num=None):
    if file[-3:]=='pkl' or file[-3:]=='csv':
        if file[-3:] == 'pkl':
            data = pd.read_pickle(PATH+file)
        else:
            data = pd.read_csv(PATH+file)
        
        if num is not None:
            num = min(num,len(data))
            data = data[0:num]
        lineList_all = data.values.tolist()
        characters = set(chain.from_iterable(lineList_all))
        lineList_all = [['START'] + w for w in lineList_all]
        x_lengths = [len(x) - Counter(x)[pad_token] for x in lineList_all]
    else:
        lineList_all = list()
        characters = list()
        x_lengths = list()
        count = 0
        with open(PATH+file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line.strip()
                lineList = list(line)
                try:
                    lineList.remove('\n')
                except ValueError:
                    pass
                x_lengths.append(len(lineList) + 1)
                characters.extend(lineList)
                if len(lineList)<SEQ_LENGTH:
                    lineList.extend([pad_token] * (SEQ_LENGTH - len(lineList)))
                lineList_all.append(['START']+lineList)
                count += 1
                if num is not None and count >= num:
                    break

    vocabulary = dict([(y,x+1) for x, y in enumerate(set(characters))])
    reverse_vocab = dict([(x+1,y) for x, y in enumerate(set(characters))])
    # add start and end tag:
    vocabulary['START'] = 0
    reverse_vocab[0] = 'START'
    if pad_token not in vocabulary.keys():
        vocabulary[pad_token] = len(vocabulary)
        reverse_vocab[len(vocabulary)-1] = pad_token
    vocabulary['END'] = len(vocabulary)
    reverse_vocab[len(vocabulary)-1] = 'END'

    tmp = sorted(zip(x_lengths,lineList_all), reverse=True)
    x_lengths = [x for x,y in tmp]
    lineList_all = [y for x,y in tmp]
    generated_data = [int(vocabulary[x]) for y in lineList_all for i,x in enumerate(y) if i<SEQ_LENGTH]
    x = torch.tensor(generated_data,device=DEVICE).view(-1,SEQ_LENGTH)
    return x.int(), vocabulary, reverse_vocab, x_lengths

def decode(token_tbl, reverse_vocab, log=None):
    words_all = []
    for n in token_tbl:
        words = [reverse_vocab[int(l)] for l in n]
        words_all.append(words[1:])
        if log is not None:
            log.write(''.join(words[1:])+'\n')
    return words_all