# -*- coding: utf-8 -*-

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os,json
from utils import audio
from text import text_to_sequence


def build_from_path(hparams, in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    """
    Preprocesses the speech dataset from a gven input path to given output directories

    Args:
        - hparams: hyper parameters
        - input_dir: input directory that contains the files to prerocess
        - out_dir: output directory of npz files
        - n_jobs: Optional, number of worker process to parallelize across
        - tqdm: Optional, provides a nice progress bar

    Returns:
        - A list of tuple describing the train examples. this should be written to train.txt
    """

    executor = ProcessPoolExecutor(max_workers=num_workers) #CPU할당
    futures = []
    index = 1

    #path = os.path.join(in_dir, 'transcript.v.1.1.txt')
    
    with open(os.path.join(in_dir, 'transcript.v.1.1.txt'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir,'%s' % parts[0])
            #print(wav_path)
            # In case of test file
            if not os.path.exists(wav_path):
                continue
            text = parts[2]
            #print('===text===')
            #print(text)
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, wav_path, text, hparams)))
            #print('====futures====')
            #print(futures)
            index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None]
    #build_from_path결과
    #('1_1_0001-audio.npy', '1_1_0001-mel.npy', '1_1_0001-linear.npy', 240300, 801, '반응도 해주시고 기분도 좋고 진짜 열심히 하겠다 생각을 많이 했어요 제가 사실 개인기를 보여 달라고 해서 딱히 별다른 개인기가 없어서.', '1_1_0001.npz')
    #위 처럼 처음부터 끝까지 모든 데이터를 하나의 list에 (append니까)

#     result = []
#     for future in tqdm(futures):
#         if future.result() is not None:
#             result.append(future.result())
#          
#     return result

def _process_utterance(out_dir, wav_path, text, hparams):
    """
    Preprocesses a single utterance wav/text pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectograms into
        - linear_dir: the directory to write the linear spectrograms into
        - wav_dir: the directory to write the preprocessed wav into
        - index: the numeric index to use in the spectogram filename
        - wav_path: path to the audio file containing the speech input
        - text: text spoken in the input audio file
        - hparams: hyper parameters

    Returns:
        - A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
    """
    try:
        # Load the audio as numpy array
        wav = audio.load_wav(wav_path, sr=hparams.sample_rate) #1차원짜리 wav파일 뽑아옴
        #Load an audio file as a floating point time series.
        #Audio will be automatically resampled to the given rate (default sr=22050).
        #To preserve the native sampling rate of the file, use sr=None. 
        #print('====wav====')
        #print(wav,wav.shape) (240001,)
    except FileNotFoundError: #catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
            wav_path))
        return None

    #rescale wav
    if hparams.rescaling:   # hparams.rescale = True
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    #We rescale because it is assumed in Wavenet training that wavs are in [-1, 1] when computing the mixture loss. This is mainly coming from PixelCNN implementation.
    #https://github.com/Rayhane-mamah/Tacotron-2/issues/69

    #M-AILABS extra silence specific
    if hparams.trim_silence:  # hparams.trim_silence = True
        wav = audio.trim_silence(wav, hparams)   # Trim leading and trailing silence

    #Mu-law quantize, default 값은 'raw'
    #The quantization noise is from the analog to digital conversion. The mu-law compression actually reduces the noise and increases the dynamic range.
    #If you search a little bit in the code you will find that the input is always mu-law encoded here.
    #scalar_input only determines if the model uses a one-hot encoding for every data point of the input waveform, or just uses floating point values for each sample.
    if hparams.input_type=='mulaw-quantize':
        #[0, quantize_channels)
        out = audio.mulaw_quantize(wav, hparams.quantize_channels)

        #Trim silences
        start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
        wav = wav[start: end]
        out = out[start: end]

        constant_values = mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16

    elif hparams.input_type=='mulaw':
        #[-1, 1]
        out = audio.mulaw(wav, hparams.quantize_channels)
        constant_values = mulaw(0., hparams.quantize_channels)
        out_dtype = np.float32

    else:  # raw
        #[-1, 1]
        out = wav
        constant_values = 0.
        out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    #print('====mel_spectrogram====')
    #print(mel_spectrogram,mel_spectrogram.shape) #(80,797),(80,801) ...
    mel_frames = mel_spectrogram.shape[1]
    #print('===mel frame====')
    #print(mel_frames) 801, 797 ,...
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:   # hparams.max_mel_frames = 1000, hparams.clip_mels_length = True
        return None

    #Compute the linear scale spectrogram from the wav
    linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
    #print('====linear_spectrogram====')
    #print(linear_spectrogram,linear_spectrogram.shape) #(1025,787),(1025,801)
    linear_frames = linear_spectrogram.shape[1]

    #sanity check
    assert linear_frames == mel_frames

    if hparams.use_lws:    # hparams.use_lws = False
        #Ensure time resolution adjustement between audio and mel-spectrogram
        fft_size = hparams.fft_size if hparams.win_size is None else hparams.win_size
        l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

        #Zero pad audio signal
        out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    else:
        #Ensure time resolution adjustement between audio and mel-spectrogram
        pad = audio.librosa_pad_lr(wav, hparams.fft_size, audio.get_hop_size(hparams)) #1024 == 2048//2 == fft_size//2
        #print('===pad===')
        #print(pad) 
        #Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
        #print(out,out.shape) #(240001,)
        out = np.pad(out, pad, mode='reflect') #shape : (242049,) - 패딩
        #print(out,out.shape) #(242049,)
        #print('===out====')
        #print(out,out.shape)

    assert len(out) >= mel_frames * audio.get_hop_size(hparams)

    #time resolution adjustement
    #ensure length of raw audio is multiple of hop size so that we can use
    #transposed convolution to upsample
    out = out[:mel_frames * audio.get_hop_size(hparams)] #240300으로 맞춤(자름)
    assert len(out) % audio.get_hop_size(hparams) == 0
    time_steps = len(out)
    #print(audio.get_hop_size(hparams)) : 300
    #print(out,out.shape) #(240300,) = 801*300
    # Write the spectrogram and audio to disk
    wav_id = os.path.splitext(os.path.basename(wav_path))[0] #확장자 제외하고 파일 이름 얻기
    #print('====wav_id====')
    #print(wav_id)
    # Write the spectrograms to disk:
    audio_filename = '{}-audio.npy'.format(wav_id)
    mel_filename = '{}-mel.npy'.format(wav_id)
    linear_filename = '{}-linear.npy'.format(wav_id)
    npz_filename = '{}.npz'.format(wav_id)
    npz_flag=True
    if npz_flag:
        # Tacotron 코드와 맞추기 위해, 같은 key를 사용한다.
        data = {
            'audio': out.astype(out_dtype),
            'mel': mel_spectrogram.T,  
            'linear': linear_spectrogram.T,
            'time_steps': time_steps,
            'mel_frames': mel_frames,
            'text': text,
            'tokens': text_to_sequence(text),   # eos(~)에 해당하는 "1"이 끝에 붙는다.
            'loss_coeff': 1  # For Tacotron
        }
        #print('=====data====')
        #print(data)
        np.savez(os.path.join(out_dir,npz_filename ), **data, allow_pickle=False) #여러개의 배열을 1개의 압축되지 않은 *.npz 포맷 파일로 저장하기
    else:
        np.save(os.path.join(out_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
        np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
        np.save(os.path.join(out_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example
    #print('====mel_frames====')
    #print(mel_frames)
    #print('====time_steps====')
    #print(time_steps)
    return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text,npz_filename)