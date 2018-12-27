# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf


def encoder(inputs, is_training=True, scope="encoder", reuse=None): #text를 숫자로 변형 (prenet -> CBHG(1-D Convolution Bank + Highway network + Bidirectional GRU))
    #입력문자 embedding열을 받아 annotation vector를 출력하는 부분
    '''
    Args:
      inputs: A 2d tensor with shape of [N, T_x, E], with dtype of int32. Encoder inputs.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:됨
      A collection of Hidden vectors. So-called memory. Has the shape of (N, T_x, E).
    '''
    with tf.variable_scope(scope, reuse=reuse): 
        # Encoder pre-net(dropout 기법을 적용한 2층의 fully connected layer로서 overfitting을 방지하고 training이 수렴하는걸 도움)
        prenet_out = prenet(inputs, is_training=is_training) # (N, T_x, E/2)
        
        # Encoder CBHG(보다 robust한 인코더를 위해 RNN이 아닌 CBHG사용, 그 전에는 입력 문자 임베딩 열이 pre-net을 거치도록함)
        ## Conv1D banks
        enc = conv1d_banks(prenet_out, K=hp.encoder_num_banks, is_training=is_training) # (N, T_x, K*E/2) / K = 5
        #unigram부터 K-gram까지를 모델링하기 위해 1부터 K까지의 길이를 가ㅈ는 필터로 입력을 convolution하고, 그 결과들을 쌓음
        
        ## Max pooling - 문맥이 달라져도 변하지 않는 부분들을 가ㅇ조 (local invariance를 키우기 위해) / t 시간축 상의 해상도를 유지하기 위해 stride = 1
        enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding="same")  # (N, T_x, K*E/2) 
          
        ## Conv1D projections - high level feature들을 뽑기 위해 몇층의 1차 convolution을 거친 뒤 highway network까지 거치도록 함
        enc = conv1d(enc, filters=hp.embed_size//2, size=3, scope="conv1d_1") # (N, T_x, E/2)
        enc = bn(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")
        #모든 1차 conv는 batch norm을 함께 사용하여 internal covariate shift문제 해결

        enc = conv1d(enc, filters=hp.embed_size // 2, size=3, scope="conv1d_2")  # (N, T_x, E/2)
        enc = bn(enc, is_training=is_training, scope="conv1d_2")

        enc += prenet_out # (N, T_x, E/2) # residual connections(1차 convolution들의 결과에 처음 입력을 더한 value가 highway network의 입력으로 들어가게 됨)
        #residual connections : 깊은 구조에서 훈련의 수렴을 도움
          
        ## Highway Nets
        for i in range(hp.num_highwaynet_blocks):
            enc = highwaynet(enc, num_units=hp.embed_size//2, 
                                 scope='highwaynet_{}'.format(i)) # (N, T_x, E/2)

        ## Bidirectional GRU - highway network의 결과가 최종적으로 bidirectional GRU의 입력이 됨
        memory = gru(enc, num_units=hp.embed_size//2, bidirection=True) # (N, T_x, E)
        #Bidirectional GRU의 forward annotation vector와 backward annotation vector를 연결한 벡터가 최종적으로 입력의 annotation vector가 됨
    return memory
        
def decoder1(inputs, memory, is_training=True, scope="decoder1", reuse=None):
    #특정 time step frame의 spectrogram을 입력으로 받고, 다음 step frame의 spectrogram을 출력
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels(*r)]. Shifted log melspectrogram of sound files.
      memory: A 3d tensor with shape of [N, T_x, E].
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted log melspectrogram tensor with shape of [N, T_y/r, n_mels*r].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        inputs = prenet(inputs, is_training=is_training)  # (N, T_y/r, E/2)
        #print('=====inputs_docoder=====')
        #print(inputs.shape)#(16,?,64)
    
        # Attention RNN
        dec, state = attention_decoder(inputs, memory, num_units=hp.embed_size) # (N, T_y/r, E)
        
        ## for attention monitoring
        alignments = tf.transpose(state.alignment_history.stack(),[1,2,0])력
        #print('-----alignment-----')
        #print(alignments.shape) #(16,?,?)
        # Decoder RNNs
        dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru1") # (N, T_y/r, E) / residual connections
        dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru2") # (N, T_y/r, E)
        #print('------dec-----')
        #print(dec.shape) #(16,?,128)
        # Outputs => (N, T_y/r, n_mels*r)
        mel_hats = tf.layers.dense(dec, hp.n_mels*hp.r) #(16,?,400)
        #print('======mel_hats=====')
        #print(mel_hats.shape)
    return mel_hats, alignments
    #80밴드의 mel scale spectrogram으로 사용(바로 linear scale spectrogram과 같은 고차원 표적을 설정할 경우 연산량이 많아지고, 필요 이상의 정보가 많아 정밀한 디코딩이 어렵다)


def decoder2(inputs, is_training=True, scope="decoder2", reuse=None):
    #decoder1의 출력이 mel scale이므로 이를 linear scale로 변환하기 위해 후처리 network 사용(decoder2)
    #decoder1의 출력을 모든 time step에 대해 고려할 수 있다는 장점을 가짐
    '''Decoder Post-processing net = CBHG
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels*r]. Log magnitude spectrogram of sound files.
        It is recovered to its original shape.
      is_training: Whether or not the layer is in training mode.  
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted linear spectrogram tensor with shape of [N, T_y, 1+n_fft//2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Restore shape -> (N, Ty, n_mels)
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, hp.n_mels])

        # Conv1D bank
        dec = conv1d_banks(inputs, K=hp.decoder_num_banks, is_training=is_training) # (N, T_y, E*K/2)
         
        # Max pooling
        dec = tf.layers.max_pooling1d(dec, pool_size=2, strides=1, padding="same") # (N, T_y, E*K/2)

        ## Conv1D projections
        dec = conv1d(dec, filters=hp.embed_size // 2, size=3, scope="conv1d_1")  # (N, T_x, E/2)
        dec = bn(dec, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

        dec = conv1d(dec, filters=hp.n_mels, size=3, scope="conv1d_2")  # (N, T_x, E/2)
        dec = bn(dec, is_training=is_training, scope="conv1d_2")

        # Extra affine transformation for dimensionality sync
        dec = tf.layers.dense(dec, hp.embed_size//2) # (N, T_y, E/2)
         
        # Highway Nets(후처리 네트워크로 highway network 사용)
        for i in range(4):
            dec = highwaynet(dec, num_units=hp.embed_size//2, 
                                 scope='highwaynet_{}'.format(i)) # (N, T_y, E/2)
         
        # Bidirectional GRU    
        dec = gru(dec, hp.embed_size//2, bidirection=True) # (N, T_y, E)
        
        # Outputs => (N, T_y, 1+n_fft//2)
        outputs = tf.layers.dense(dec, 1+hp.n_fft//2)

    return outputs #linear spectrogram을 출력
    #음성 신호를 합성하기 위해 1025차 linear scale spectrogram으로 변환한 뒤 최종적으로 음성 신호 출력
