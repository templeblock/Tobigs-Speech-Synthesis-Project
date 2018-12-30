# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

import os
from hyperparams import Hyperparams as hp
import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch, load_vocab
from modules import *
from networks import encoder, decoder1, decoder2
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class Graph:
    def __init__(self, mode="train"):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        # Set phase
        is_training=True if mode=="train" else False

        # Graph
        # Data Feeding
        # x: Text. (N, Tx)
        # y: Reduced melspectrogram. (N, Ty//r, n_mels*r)
        # z: Magnitude. (N, Ty, n_fft//2+1)
        if mode=="train":
            self.x, self.y, self.z, self.fnames, self.num_batch = get_batch()
        elif mode=="eval":
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels*hp.r))
            self.z = tf.placeholder(tf.float32, shape=(None, None, 1+hp.n_fft//2))
            self.fnames = tf.placeholder(tf.string, shape=(None,))
        else: # Synthesize
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels * hp.r))
        #print('=====y_shape====')
        #print(self.y.shape)
        #self.y = tf.reshape(self.y,[hp.batch_size,None,hp.n_mels * hp.r])
        #print('=====y_shape====')
        #print(self.y.shape)
        # Get encoder/decoder inputs
        self.encoder_inputs = embed(self.x, len(hp.vocab), hp.embed_size) # (N, T_x, E)
        self.decoder_inputs = tf.concat((tf.zeros_like(self.y[:, :1, :]), self.y[:, :-1, :]), 1) # (N, Ty/r, n_mels*r), shape: (16,?,80)
        self.decoder_inputs = self.decoder_inputs[:, :, -hp.n_mels:] # feed last frames only (N, Ty/r, n_mels)
        #print('==================decoder_inputs================')
        #print(self.decoder_inputs.shape) #(16,?,80)

        # Networks
        with tf.variable_scope("net"):
            # Encoder
            self.memory = encoder(self.encoder_inputs, is_training=is_training) # (N, T_x, E)
            #print('memory_shape')
            #print(self.memory.shape) #(16,?,128-embeding_size)
            # Decoder1
            self.y_hat, self.alignments = decoder1(self.decoder_inputs,
                                                     self.memory,
                                                     is_training=is_training) # (N, T_y//r, n_mels*r)
            # Decoder2 or postprocessing
            self.z_hat = decoder2(self.y_hat, is_training=is_training) # (N, T_y//r, (1+n_fft//2)*r)
        #print('===z_hat===')
        #print(self.z_hat.shape)
        # monitor
        self.audio = tf.py_func(spectrogram2wav, [self.z_hat[0]], tf.float32) 
        #decoder2를 통해 나온 linear spectrogram을 음성신호로 합성할때 Griffin Lim(=spectrogram2wav)사용
        #반복적인 과정을 통해 주어진 modified STFT magnitude와 가장 비슷한 STFT magnitude을 가진 음성 기호를 복원하는 알고리즘
            #1. 이전 단계에서 출력된 음성신호의 STFT를 계산한 뒤 진폭을 입력으로 주어진 MSTFTM으로 대체
            #2. 새로운 STFT의 진폭과 입력 MSTFT의 진폭이 squared error가 최소가 되도록 원래 신호를 복원
            #1.,2. iter
        if mode in ("train", "eval"):
            # Loss
            #print('shape???')
            #print(self.y_hat.shape) #(16,?,400)
            #print(self.y.shape) #(16,?,80)
            self.loss1 = tf.reduce_mean(tf.abs(self.y_hat - self.y)) #decoder의 mel scale spectrogram의 L1 loss            
            self.loss2 = tf.reduce_mean(tf.abs(self.z_hat - self.z)) #후처리 네트워크의 linear scale spectrogram의 L1 loss
            self.loss = self.loss1 + self.loss2 #가중치 합
            #Each L1 Loss의 가중치는 같고, linear scale spectrogram의 L1 loss는 3000 Hz이하의 값들에 대해 가중치를 둬서 사용함
            #(L2 / L1 에서 5000Hz이하 값들에 가중치를 둘때보다 성능 좋음)

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.lr = learning_rate_decay(hp.lr, global_step=self.global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            ## gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_norm(grad, 5.)
                self.clipped.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

            # Summary
            tf.summary.scalar('{}/loss1'.format(mode), self.loss1)
            tf.summary.scalar('{}/loss'.format(mode), self.loss)
            tf.summary.scalar('{}/lr'.format(mode), self.lr)

            tf.summary.image("{}/mel_gt".format(mode), tf.expand_dims(self.y, -1), max_outputs=1)
            tf.summary.image("{}/mel_hat".format(mode), tf.expand_dims(self.y_hat, -1), max_outputs=1)
            tf.summary.image("{}/mag_gt".format(mode), tf.expand_dims(self.z, -1), max_outputs=1)
            tf.summary.image("{}/mag_hat".format(mode), tf.expand_dims(self.z_hat, -1), max_outputs=1)

            tf.summary.audio("{}/sample".format(mode), tf.expand_dims(self.audio, 0), hp.sr)
            self.merged = tf.summary.merge_all()
         
if __name__ == '__main__':
    g = Graph(); print("Training Graph loaded")
    
    #그래프 해석 : https://github.com/keithito/tacotron/issues/144
    #gpu_fraction = 0.1
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))    
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) 
    
    #with g.graph.as_default():
    sv = tf.train.Supervisor(logdir=hp.logdir, save_summaries_secs=60, save_model_secs=0)
    # supervisor는 세션 초기화를 관리하고, checkpoint로부터 모델을 복원하고 
    # 에러가 발생하거나 연산이 완료되면 프로그램을 종료.
    with sv.managed_session() as sess:
        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                _, gs = sess.run([g.train_op, g.global_step])

                # Write checkpoint files
                if gs % 1000 == 0:
                    sv.saver.save(sess, hp.logdir + '/model_gs_{}k'.format(gs//1000))

                    # plot the first alignment for logging
                    al = sess.run(g.alignments)
                    #print('====al=====')
                    #print(al[0])
                    plot_alignment(al[0], gs)
            if gs >hp.num_iterations: break

    print("Done")
