# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:46:04 2018
@author: natnij

Based on SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient, 
    Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu.
    Paper available here: https://arxiv.org/abs/1609.05473
Translated from the original tensorflow repo: 
    https://github.com/LantaoYu/SeqGAN, and adjusted for wider usability.
Many thanks to the original authors.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal

class Embedding(nn.Module):
    def __init__(self, vocab_size, emb_size, stdDev=0.1):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.normalDistr = Normal(torch.tensor([0.0]), torch.tensor([stdDev]))
        normalSample = self.normalDistr.sample(
                        torch.Size([vocab_size, emb_size])).squeeze(dim=2)
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.embedding.weight.data.copy_(normalSample)
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = len(x)
        x = self.embedding(x.long()).view([-1, 1, seq_len, self.emb_size])
        return x