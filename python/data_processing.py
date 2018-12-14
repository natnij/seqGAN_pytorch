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
import torch
from config import SEQ_LENGTH,VOCAB_SIZE,GENERATE_NUM,DEVICE,PATH

def gen_record(num=GENERATE_NUM):
    # batch x nChannels x Height x Width
    data = torch.rand(num, SEQ_LENGTH-1, device=DEVICE)
    data = torch.abs(data * (VOCAB_SIZE-2)).int()+1
    data = torch.cat([torch.zeros([num,1]).int(), data], dim=1)
    return data

def gen_label(num=GENERATE_NUM, target_space=2, fixed_value=None):
    # the loss function's target should be a torch.LongTensor.
    # the target's dimension should be only 1D with class index (target_space).
    if fixed_value is None:
        return torch.randint(low=0, high=target_space, size=(num,), device=DEVICE).long()
    else:
        assert fixed_value < target_space
        return torch.randint(low=fixed_value, high=fixed_value+1, size=(num,), device=DEVICE).long()

def read_sampleFile(file='real_data_chinesePoems.txt'):
    lineList_all = list()
    characters = list()
    x_lengths = list()
    with open(PATH+file, 'r') as f:
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
                lineList.extend(['PAD'] * (SEQ_LENGTH - len(lineList)))
            lineList_all.append(['START']+lineList)
    vocabulary = dict([(y,x+1) for x, y in enumerate(set(characters))])
    reverse_vocab = dict([(x+1,y) for x, y in enumerate(set(characters))])
    # add start and end tag:
    vocabulary['START'] = 0
    reverse_vocab[0] = 'START'
    vocabulary['PAD'] = len(vocabulary)
    reverse_vocab[len(vocabulary)-1] = 'PAD'
    vocabulary['END'] = len(vocabulary)
    reverse_vocab[len(vocabulary)-1] = 'END'

    tmp = sorted(zip(x_lengths,lineList_all), reverse=True)
    x_lengths = [x for x,y in tmp]
    lineList_all = [y for x,y in tmp]
    generated_data = [int(vocabulary[x]) for y in lineList_all for i,x in enumerate(y) if i<SEQ_LENGTH]
    x = torch.Tensor(generated_data).view(-1,SEQ_LENGTH)
    return x.int(), vocabulary, reverse_vocab, x_lengths

def decode(token_tbl, reverse_vocab, log=None):
    words_all = []
    for n in token_tbl:
        words = [reverse_vocab[int(l)] for l in n]
        words_all.append(words[1:])
        if log is not None:
            log.write(''.join(words[1:])+'\n')
    return words_all