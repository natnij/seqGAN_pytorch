# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:21:03 2018
"""
from itertools import chain
import jieba
import pandas as pd
from config import PATH, SEQ_LENGTH

def wordseg(x, pad_token='PAD'):
    try:
        text = list(jieba.cut(x))
        text = [w for w in text if w not in ['\n',' ']]
    except:
        text = []
    if len(text) <= SEQ_LENGTH-1:
        text = text + [pad_token] * (SEQ_LENGTH - 1 - len(text))
    return pd.Series(text[0:SEQ_LENGTH-1])

def delSpace(x, ignored=[' ','\n','|','*']):
    for t in ignored:
        x = x.replace(t, '')
    return x

def splitSentence(x, splitBy=['。', '！','；','……']):
    tmp = [x]
    for t in splitBy:
        tmp = [s.split(t) for s in tmp]
        if isinstance(tmp[0],list):
            tmp = list(chain.from_iterable(tmp))
    tmp = [x for x in tmp if len(x)>5]
    return tmp

def readRandomText(inputFilename='london.txt',outputFilename='real_data_london.pkl'):
    lineList_all = list()
    with open(PATH+inputFilename, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line.strip()
            lineList_all.append(line)

    data = [delSpace(x) for x in lineList_all if len(x) > 5]
    data = pd.DataFrame(list(chain.from_iterable([splitSentence(x) for x in data])))
    data.columns = ['data']
    sentences = data.apply(lambda row: wordseg(row['data']),axis=1)
    coln = ['token'+str(x) for x in sentences.columns.tolist()]
    sentences.columns = coln
    sentences.to_pickle(PATH+outputFilename)
    return sentences

#%%
if __name__ == '__main__':

    sentences = readRandomText(inputFilename='london.txt',outputFilename='real_data_london.pkl')

