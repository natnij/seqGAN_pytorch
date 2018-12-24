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
SEQ_LENGTH = 21 # x: 'START' + tokens; y: tokens + 'END'
EMB_SIZE = 32
GENERATE_NUM = 1000
FILTER_SIZE = list(range(1,SEQ_LENGTH))
NUM_FILTER =  ([100] + [200] * 9 + [160] * SEQ_LENGTH)[0:SEQ_LENGTH-1]
DIS_NUM_EPOCH = 50
DIS_NUM_EPOCH_PRETRAIN = 100
GEN_NUM_EPOCH = 50
GEN_NUM_EPOCH_PRETRAIN = 100
GEN_HIDDEN_DIM = 48
ROLLOUT_ITER = 16
TOTAL_BATCH = 20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
#DEVICE = torch.device('cpu')
#DTYPE = torch.FloatTensor
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
    log.write('\nAllocated:'+str(round(torch.cuda.memory_allocated(0)/1024**3,1))+'GB')
    log.write('\nCached:   '+str(round(torch.cuda.memory_cached(0)/1024**3,1))+'GB')
    log.close()
