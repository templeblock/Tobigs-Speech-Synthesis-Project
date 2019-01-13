# coding: utf-8
# Code based on https://github.com/keithito/tacotron/blob/master/models/tacotron.py

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.layers import core
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _bahdanau_score, _BaseAttentionMechanism, BahdanauAttention, AttentionWrapper, AttentionWrapperState


def get_embed(inputs, num_inputs, embed_size, name):  # speaker_id, self.num_speakers, hp.enc_prenet_sizes[-1], "before_highway"
    #print('====inputs====')
    #print(inputs)
    embed_table = tf.get_variable(name, [num_inputs, embed_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
    return tf.nn.embedding_lookup(embed_table, inputs)


def prenet(inputs, is_training, layer_sizes, drop_prob, scope=None):
    x = inputs  # 3차원 array(batch,seq_length,embedding_dim)   ==> (batch,seq_length,256)  ==> (batch,seq_length,128)
    drop_rate = drop_prob if is_training else 0.0
    #print('drop_rate',drop_rate)
    with tf.variable_scope(scope or 'prenet'):
        for i, size in enumerate(layer_sizes):  # [f(256), f(128)]
            dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
            x = tf.layers.dropout(dense, rate=drop_rate,training=is_training, name='dropout_%d' % (i+1))
    return x

def cbhg(inputs, input_lengths, is_training, bank_size, bank_channel_size, maxpool_width, highway_depth, 
         rnn_size, proj_sizes, proj_width, scope,before_highway=None, encoder_rnn_init_state=None):
    # inputs: (N,T_in, 128), bank_size: 16
    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):
            # Convolution bank: concatenate on the last axis
            # to stack channels from all convolutions
            conv_fn = lambda k: conv1d(inputs, k, bank_channel_size, tf.nn.relu, is_training, 'conv1d_%d' % k)  # bank_channel_size =128

            conv_outputs = tf.concat( [conv_fn(k) for k in range(1, bank_size+1)], axis=-1,)  # ==> (N,T_in,128*bank_size)
            #unigram부터 K-gram까지를 모델링하기 위해 1부터 K까지의 길이를 가ㅈ는 필터로 입력을 convolution하고, 그 결과들을 쌓음

        # Maxpooling:문맥이 달라져도 변하지 않는 부분들을 가ㅇ조 (local invariance를 키우기 위해) / t 시간축 상의 해상도를 유지하기 위해 stride = 1
        maxpool_output = tf.layers.max_pooling1d(conv_outputs,pool_size=maxpool_width,strides=1,padding='same')  # maxpool_width = 2

        # Two projection layers:
        # Conv1D projections - high level feature들을 뽑기 위해 몇층의 1차 convolution을 거친 뒤 highway network까지 거치도록 함
        #모든 1차 conv는 batch norm을 함께 사용하여 internal covariate shift문제 해결
        proj_out = maxpool_output
        for idx, proj_size in enumerate(proj_sizes):   # [f(128), f(128)],  post: [f(256), f(80)]
            activation_fn = None if idx == len(proj_sizes) - 1 else tf.nn.relu
            #print(proj_width) # 3
            proj_out = conv1d(proj_out, proj_width, proj_size, activation_fn,is_training, 'proj_{}'.format(idx + 1))  # proj_width = 3

        # Residual connection:
        # residual connections(1차 convolution들의 결과에 처음 입력을 더한 value가 highway network의 입력으로 들어가게 됨)
        # residual connections : 깊은 구조에서 훈련의 수렴을 도움
        if before_highway is not None: # multi-sperker mode
            expanded_before_highway = tf.expand_dims(before_highway, [1])
            tiled_before_highway = tf.tile(expanded_before_highway, [1, tf.shape(proj_out)[1], 1])

            highway_input = proj_out + inputs + tiled_before_highway
        else: # single model
            highway_input = proj_out + inputs

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != rnn_size:  # rnn_size = 128
            highway_input = tf.layers.dense(highway_input, rnn_size)

        # 4-layer HighwayNet:
        for idx in range(highway_depth): # 4
            highway_input = highwaynet(highway_input, 'highway_%d' % (idx+1))

        rnn_input = highway_input # Bidirectional GRU - highway network의 결과가 최종적으로 bidirectional GRU의 입력이 됨

        # Bidirectional RNN
        if encoder_rnn_init_state is not None:
            initial_state_fw, initial_state_bw = tf.split(encoder_rnn_init_state, 2, 1)
        else:  # single mode
            initial_state_fw, initial_state_bw = None, None

        cell_fw, cell_bw = GRUCell(rnn_size), GRUCell(rnn_size)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,rnn_input,sequence_length=input_lengths,
                                                          initial_state_fw=initial_state_fw,initial_state_bw=initial_state_bw,dtype=tf.float32)
        return tf.concat(outputs, axis=2)    # Concat forward and backward


def batch_tile(tensor, batch_size):
    expaneded_tensor = tf.expand_dims(tensor, [0])
    return tf.tile(expaneded_tensor, \
            [batch_size] + [1 for _ in tensor.get_shape()])


def highwaynet(inputs, scope):
    highway_dim = int(inputs.get_shape()[-1])

    with tf.variable_scope(scope):
        H = tf.layers.dense(inputs,units=highway_dim, activation=tf.nn.relu,name='H')
        T = tf.layers.dense(inputs,units=highway_dim, activation=tf.nn.sigmoid,name='T',bias_initializer=tf.constant_initializer(-1.0))
        #return : 한 층의 신경망을 거친 결과, H와 원래 입력 두 값을 weighted sum하는 구조(일반화에 도움이됨)
        return H * T + inputs * (1.0 - T)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope): #kernel_size = filter_size
    with tf.variable_scope(scope):
        # strides=1, padding = same 이므로, kernel_size에 상관없이 크기가 유지된다.
        conv1d_output = tf.layers.conv1d(inputs,filters=channels,kernel_size=kernel_size,activation=activation,padding='same') # padding이 same이라 kenel size가 달라도 concat된다.
        return tf.layers.batch_normalization(conv1d_output, training=is_training) #모든 1차 conv는 batch norm을 함께 사용하여 internal covariate shift문제 해결

