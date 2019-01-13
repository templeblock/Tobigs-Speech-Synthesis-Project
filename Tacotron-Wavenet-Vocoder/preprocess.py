# coding: utf-8
"""
python preprocess.py --num_workers 10 --name son --in_dir D:\hccho\multi-speaker-tacotron-tensorflow-master\datasets\son --out_dir .\data\son
python preprocess.py --num_workers 10 --name moon --in_dir D:\hccho\multi-speaker-tacotron-tensorflow-master\datasets\moon --out_dir .\data\moon
 ==> out_dir에 LJ001-0001-audio.npy, LJ001-0001-mel.npy 들이 생성된다.
 
 
 
"""
import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
import importlib
from hparams import hparams, hparams_debug_string
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def preprocess(mod, in_dir, out_root,num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = mod.build_from_path(hparams, in_dir, out_dir,num_workers=num_workers, tqdm=tqdm)
    #('1_1_0001-audio.npy', '1_1_0001-mel.npy', '1_1_0001-linear.npy', 240300, 801, '반응도 해주시고 기분도 좋고 진짜 열심히 하겠다 생각을 많이 했어요 제가 사실 개인기를 보여 달라고 해서 딱히 별다른 개인기가 없어서.', '1_1_0001.npz')
    #print('===metadata===')
    #print(metadata)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir): #train.txt파일 생성함수
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    #m : ('1_1_0013-audio.npy', '1_1_0013-mel.npy', '1_1_0013-linear.npy', 240300, 801, '정말 너무 너무 보고싶고 김연아 선수도 너무 요즘에 너무 정말 떨어지는게 없잖아요. 너무 예쁘시고 운동도 잘 하시고 노래도 노래까지 잘하시고.', '1_1_0013.npz')
    
    mel_frames = sum([int(m[4]) for m in metadata])
    #print(mel_frames) 10310 (싹다 더한거)
    timesteps = sum([int(m[3]) for m in metadata])
    #print(timesteps) 240300 (싹다 더한거)
    sr = hparams.sample_rate
    hours = timesteps / sr / 3600
    print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(len(metadata), mel_frames, timesteps, hours))
    print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
    print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
    print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))
    
    #Write 13 utterances, 10310 mel frames, 3093000 audio timesteps, (0.04 hours) 
    #Max input length (text chars): 95 - 자모단위로 잘랐을때 char 최댓값
    #Max mel frames length: 801
    #Max audio timesteps length: 240300

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--in_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=str, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    args = parser.parse_args()

    if args.hparams is not None:
        hparams.parse(args.hparams)
    print(hparams_debug_string())

    name = args.name
    in_dir = args.in_dir
    out_dir = args.out_dir
    num_workers = args.num_workers
    num_workers = cpu_count() if num_workers is None else int(num_workers)  # cpu_count() = process 갯수

    print("Sampling frequency: {}".format(hparams.sample_rate))

    assert name in ["cmu_arctic", "ljspeech", "son", "moon","kss",'IU','TY']
    mod = importlib.import_module('datasets.{}'.format(name)) #mod라는 변수를  통해 module안에 class 호출이 가능
    #print(mod) <module 'datasets.IU' from '/home/jeon/Desktop/Speech_project/Tacotron-Wavenet-Vocoder/datasets/IU.py'>
    preprocess(mod, in_dir, out_dir, num_workers)
