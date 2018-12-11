# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 08:52:48 2018
@author: natnij

Based on SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient, 
    Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu.
    Paper available here: https://arxiv.org/abs/1609.05473
Translated from the original tensorflow repo: 
    https://github.com/LantaoYu/SeqGAN, and adjusted for wider usability.
Many thanks to the original authors.
"""
import torch
import os
from datetime import datetime

PATH = '../data/'
MAXINT = 10000
SEQ_LENGTH = 8 # x: 'START' + tokens; y: tokens + 'END'
EMB_SIZE = 32
BATCH_SIZE = 5
GENERATE_NUM = 100
FILTER_SIZE = [  1,  2,  3,  4,  5,  6,  7,  8]
NUM_FILTER =  [100,200,200,200,160,160,160,100]
DIS_NUM_EPOCH = 5
DIS_NUM_EPOCH_PRETRAIN = 5
GEN_NUM_EPOCH = 5
GEN_NUM_EPOCH_PRETRAIN = 5
VOCAB_SIZE = 475 # num_tokens + 2
GEN_HIDDEN_DIM = 48
ROLLOUT_ITER = 16
TOTAL_BATCH = 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if DEVICE.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def openLog(filename='record.txt'):
    if os.path.exists(PATH+filename):
        append_write = 'a'
    else:
        append_write = 'w'
    log = open(PATH+filename, append_write)
    return log

if DEVICE.type == 'cuda':
    log = openLog('gpu.txt')
    log.write('datetime:{}, device name:{}\n'.format(datetime.now(), 
                                          torch.cuda.get_device_name(0)))
    log.write('Memory Usage:')
    log.write('\nAllocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    log.write('\nCached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    log.close()
    
def createOneHotDummy(dim=(BATCH_SIZE * SEQ_LENGTH, VOCAB_SIZE)):
    one_hot = torch.Tensor(dim[0],dim[1], device=DEVICE)
    return one_hot

ONEHOT_BATCH = createOneHotDummy(dim=(BATCH_SIZE * SEQ_LENGTH, VOCAB_SIZE))
ONEHOT_SINGLE = createOneHotDummy(dim=(SEQ_LENGTH, VOCAB_SIZE))
