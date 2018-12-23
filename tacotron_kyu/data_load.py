# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata
from itertools import chain
from g2p import runKoG2P

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode="train"):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # load conversion dictionaries
    j2hcj, j2sj, j2shcj = load_j2hcj(), load_j2sj(), load_j2shcj()

    #if mode in ("train","eval"): #if mode == "train"에서 바꿈!
    if mode=="train":
        # Parse
        fpaths, text_lengths, texts = [], [], []
        transcript = os.path.join(hp.data, 'transcript.v.1.1.txt')
        lines = codecs.open(transcript, 'rb', 'utf-8').readlines()
        for line in lines:
            fname, _, expanded, text, _ = line.strip().split("|")
            fpath = os.path.join(hp.data, fname)
            fpaths.append(fpath)

            if hp.num_exp==0:
                text = expanded + u"␃"  # ␃: EOS
                text = runKoG2P(text, "rulebook.txt")
            else:
                text += u"␃"  # ␃: EOS
                if hp.num_exp==2:
                    text = [j2hcj[char] for char in text]
                elif hp.num_exp==3:
                    text = [j2sj[char] for char in text]
                elif hp.num_exp==4:
                    text = [j2shcj[char] for char in text]
                text = chain.from_iterable(text)

            text = [char2idx[char] for char in text]
            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32).tostring())

        return fpaths, text_lengths, texts
    else: # synthesize on unseen test text.
        # Parse
        def _normalize(line):
            #print('=====line=====')
            #print(line.strip().split("|"))
            _, expanded, text= line.strip().split("|")
            
            if hp.num_exp==0:
                text = expanded + u"␃"  # ␃: EOS
                text = runKoG2P(text, "rulebook.txt")
            else:
                text += u"␃"
                if hp.num_exp==2:
                    text = [j2hcj[char] for char in text]
                elif hp.num_exp==3:
                    text = [j2sj[char] for char in text]
                elif hp.num_exp==4:
                    text = [j2shcj[char] for char in text]
                text = chain.from_iterable(text)
            text = [char2idx[char] for char in text]
            return text

        lines = codecs.open(hp.test_data, 'rb', 'utf8').read().splitlines()
        sents = [_normalize(line) for line in lines[1:]]
        #print('=====len_sents=====')
        #print(len(sents))
        texts = np.zeros((len(sents), hp.max_N), np.int32) #max_N :123 (12,123)
        #print("=====texts===")
        #print(texts,texts.shape)
        for i, sent in enumerate(sents):
            #print(i,sent)
            texts[i, :len(sent)] = sent
        print("====texts====")
        print(texts)
        return texts

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data() # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // hp.batch_size

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        def _load_spectrograms(fpath):
            fname = os.path.basename(fpath)
            mel = "mels/{}".format(str(fname).replace("wav", "npy").replace("/b'","")).replace("'","").replace("b","")
            mag = "mags/{}".format(str(fname).replace("wav", "npy").replace("/b'","")).replace("'","").replace("b","")
            #print('=====mel=====')
            #print(mel)
            #print('=====mag=====')
            #print(mag)
            return fname, np.load(mel), np.load(mag)

        fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, hp.n_mels*hp.r)) #hp.n_mels->hp,n_mels*hp.r (matching with y_hat)
        mag.set_shape((None, hp.n_fft//2+1))

        # Batching
        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=hp.batch_size,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=8,
                                            capacity=hp.batch_size*4,
                                            dynamic_pad=True)

    return texts, mels, mags, fnames, num_batch

