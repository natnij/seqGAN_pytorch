## SeqGAN in pytorch


### Requirements
pytorch 0.4

### Background
Written based on [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473), - *Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu.*


Translated from the original [tensorflow repo](https://github.com/LantaoYu/SeqGAN), and adjusted for wider usability.

Many thanks to the original authors.

### Inputs and outputs

Input text as shown in sample data. 

For training: input sequence x is 'START' + tokens, output sequence y is tokens + 'END'.

To generate text: run sequenceGAN.py. The generated text will be saved in genTxt.txt file.

### Params

- `SEQ_LENGTH`: uniform length of sentences. Flexible input sequence length is not tested. Default to 8 because sample Chinese poem data has 7 characters per line, plus the START token. 
- `EMB_SIZE`: embedded size, size of word2vec. 
- `BATCH_SIZE`: training batch size.
- `GENERATE_NUM`: number of sentences to be generated for training.
- `FILTER_SIZE` = [  1,  2,  3,  4,  5,  6,  7,  8]: different sizes of filters used in the discriminator. equivalent to n-grams. 
- `NUM_FILTER` =  [100,200,200,200,160,160,160,100]: corresponding filter size (or number of features to be extracted).
- `DIS_NUM_EPOCH`: number of epochs for discriminator training
- `DIS_NUM_EPOCH_PRETRAIN`: number of epochs for discriminator pretraining
- `GEN_NUM_EPOCH`: number of epochs for generator training
- `GEN_NUM_EPOCH_PRETRAIN`: number of epochs for generator pretraining
- `VOCAB_SIZE`: size of vocabulary. Is equal to actual number of tokens plus the 'START' and 'END' tokens.
- `GEN_HIDDEN_DIM`: generator (LSTM) hidden dimension.
- `ROLLOUT_ITER`: number of iterations to roll out, for calculation of rewards
- `TOTAL_BATCH`: number of batches in adversarial training
- `MAXINT`: used as penalty factor to avoid choosing start and end tokens for the sequence. The penalty factor is applied before calculating the probability of output tokens. 

### Intro to SeqGAN

For basic understanding please refer to the original [paper](https://arxiv.org/abs/1609.05473). 

What's not in the paper, is my simplified view of the world:

![alt_text](/doc/seqGAN.jpg)

The generator is not much different from a vanilla LSTM, except for a custom loss function with rewards (basically giving higher probability to generated sequences which are labeled as "real data" by the discriminator). 

The rollout module is almost a copy of the generator - if the `r_update_date` is set to 0 then it IS the exact copy. Reason why the update is set to non-zero, is explained [here](https://github.com/LantaoYu/SeqGAN/issues/29) and [here](https://arxiv.org/pdf/1509.02971.pdf). the rollout module does not participate in the gradient.

The discriminator module is a CNN with a series of parallel conv layers, acting similar to n-grams:

![alt_text](/doc/discriminator.jpg)

### Further improvements

1. The code has NOT yet been tested on GPU.

2. It does not accept flexible sequence length. 
