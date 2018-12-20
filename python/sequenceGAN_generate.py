# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:14:08 2018
"""
import sys
import torch
from config import PATH, openLog
from data_processing import decode

def main(batch_size=1):
    model = torch.load(PATH+'generator.pkl')
    reverse_vocab = torch.load(PATH+'reverse_vocab.pkl')

    num = model.generate(batch_size=batch_size)
    log = openLog('genTxt_predict.txt')
    result = decode(num, reverse_vocab, log)
    log.close()
    return result

if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1
    
    result = main(batch_size)
    print(result)
    