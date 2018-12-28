# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''
from __future__ import print_function, division

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
import librosa
import copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal
import os


def get_spectrograms(fpath):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # num = np.random.randn()
    # if num < .2:
    #     y, sr = librosa.load(fpath, sr=hp.sr)
    # else:
    #     if num < .4:
    #         tempo = 1.1
    #     elif num < .6:
    #         tempo = 1.2
    #     elif num < .8:
    #         tempo = 0.9
    #     else:
    #         tempo = 0.8
    #     cmd = "ffmpeg -i {} -y ar {} -hide_banner -loglevel panic -ac 1 -filter:a atempo={} -vn temp.wav".format(fpath, hp.sr, tempo)
    #     os.system(cmd)
    #     y, sr = librosa.load('temp.wav', sr=hp.sr)

    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)


    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels, min=hp.fmin, fmax=hp.fmax)  # (n_mels, 1+n_fft//2) / fmin, fmax추가 (noise를 줄이기 위해)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram): #vocoder(스펙트로그램들을 CBHG를 거치고 Griffin-Lim을 거쳐서 음성을 만듦)
  #decoder2를 통해 나온 linear spectrogram을 음성 신호로 합성할때 사용
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter): #griffin_lim 
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")


def plot_alignment(alignment, gs):
    """Plots the alignment
    alignments: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    gs : (int) global step
    """
    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}k.png'.format(hp.logdir, gs//1000), format='png')

def learning_rate_decay(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme from tensor2tensor'''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def load_spectrograms(fpath):
    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0 # for reduction
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    return fname, mel.reshape((-1, hp.n_mels*hp.r)), mag

def trim(wav, top_db=40, min_silence_sec=0.8):
    frame_length = int(hp.sr * min_silence_sec)
    hop_length = int(frame_length / 4)
    endpoint = librosa.effects.split(wav, frame_length=frame_length,
                               hop_length=hop_length,
                               top_db=top_db)[0, 1]
    return wav[:endpoint]

def load_j2hcj():
    '''
    Arg:
      jamo: A Hangul Jamo character(0x01100-0x011FF)

    Returns:
      A dictionary that converts jamo into Hangul Compatibility Jamo(0x03130 - 0x0318F) Character
    '''
    jamo = u'''␀␃ !,.?ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ'''
    hcj  = u'''␀␃ !,.?ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄾㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'''

    assert len(jamo) == len(hcj)
    j2hcj = {j: h for j, h in zip(jamo, hcj)}
    return j2hcj

def load_j2sj():
    '''
    Arg:
      jamo: A Hangul Jamo character(0x01100-0x011FF)

    Returns:
      A dictionary that decomposes double consonants into two single consonants.
    '''
    jamo = u'''␀␃ !,.?ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ'''
    sj = u'''␀|␃| |!|,|.|?|ᄀ|ᄀᄀ|ᄂ|ᄃ|ᄃᄃ|ᄅ|ᄆ|ᄇ|ᄇᄇ|ᄉ|ᄉᄉ|ᄋ|ᄌ|ᄌᄌ|ᄎ|ᄏ|ᄐ|ᄑ|ᄒ|ᅡ|ᅢ|ᅣ|ᅤ|ᅥ|ᅦ|ᅧ|ᅨ|ᅩ|ᅪ|ᅫ|ᅬ|ᅭ|ᅮ|ᅯ|ᅰ|ᅱ|ᅲ|ᅳ|ᅴ|ᅵ|ᆨ|ᆨᆨ|ᆨᆺ|ᆫ|ᆫᆽ|ᆫᇂ|ᆮ|ᆯ|ᆯᆨ|ᆯᆷ|ᆯᆸ|ᆯᇀ|ᆯᇂ|ᆷ|ᆸ|ᆸᆺ|ᆺ|ᆺᆺ|ᆼ|ᆽ|ᆾ|ᆿ|ᇀ|ᇁ|ᇂ'''

    assert len(jamo)==len(sj.split("|"))
    j2sj = {j: s for j, s in zip(jamo, sj.split("|"))}
    return j2sj

def load_j2shcj():
    '''
    Arg:
      jamo: A Hangul Jamo character(0x01100-0x011FF)

    Returns:
      A dictionary that converts jamo into Hangul Compatibility Jamo(0x03130 - 0x0318F) Character.
      Double consonants are further decomposed into single consonants.
    '''
    jamo = u'''␀␃ !,.?ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ'''
    shcj = u'''␀|␃| |!|,|.|?|ㄱ|ㄱㄱ|ㄴ|ㄷ|ㄷㄷ|ㄹ|ㅁ|ㅂ|ㅂㅂ|ㅅ|ㅅㅅ|ㅇ|ㅈ|ㅈㅈ|ㅊ|ㅋ|ㅌ|ㅍ|ㅎ|ㅏ|ㅐ|ㅑ|ㅒ|ㅓ|ㅔ|ㅕ|ㅖ|ㅗ|ㅘ|ㅙ|ㅚ|ㅛ|ㅜ|ㅝ|ㅞ|ㅟ|ㅠ|ㅡ|ㅢ|ㅣ|ㄱ|ㄱㄱ|ㄱㅅ|ㄴ|ㄴㅈ|ㄴㅎ|ㄷ|ㄹ|ㄹㄱ|ㄹㅁ|ㄹㅂ|ㄹㅌ|ㄹㅎ|ㅁ|ㅂ|ㅂㅅ|ㅅ|ㅅㅅ|ㅇ|ㅈ|ㅊ|ㅋ|ㅌ|ㅍ|ㅎ'''

    assert len(jamo)==len(shcj.split("|"))
    j2shcj = {j: s for j, s in zip(jamo, shcj.split("|"))}
    return j2shcj