## SeqGAN in pytorch


### Requirements
pytorch 0.4

jieba 0.39 (if you want to tokenize with wordseg.py)

### Background
Written based on [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473), - *Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu.*


Translated from the original [tensorflow repo](https://github.com/LantaoYu/SeqGAN), and adjusted for wider usability.

Many thanks to the original authors.

### Inputs and outputs

Input text as shown in sample data. Supports variable lengths input and batch processing, however for word segmentation you'd need to make some changes to the data_processing.py.

Train: 

- Input sequence length is 'START' + tokens, output sequence y is tokens + 'END'. You need to specify in `config.py` the `SEQ_LENGTH`, which is the usual embedding size plus one for the 'START' token.
- Tokenized training file can be created by running the `wordseg.py` on the original input file `london.txt`. Default training file name is `real_data_london.pkl`. 
- Training file path can be specified in `config.py`. Training file name can be set up in `data_processing.py `(currently as a default argument). 
- To train based on input file: run `sequenceGAN.py`. The generated text from training will be saved in `genTxt.txt` file. 
- To train in batches of e.g. 4: run `sequenceGAN.py 4`.
- During training, to read in only limited number of samples, e.g. 8: run `sequenceGAN.py 4 8`. If it's left out then all samples will be used for training.
- After training, the vocabulary for decoding, as well as the generator network will be saved in PATH. Training performance will be saved in `record.txt` file.

Generate new text:

- After training, to generate new text: run `sequenceGAN_generate.py`. 
- To generate more than one sample (e.g. 4 samples): run `sequenceGAN_generate.py 4`.

Test:

- To individually test the modules, run the files separately (the sanityCheck functions will be run). Result will be saved in `test.txt` file in PATH.

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

