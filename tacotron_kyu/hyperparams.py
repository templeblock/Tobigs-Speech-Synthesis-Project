# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''
class Hyperparams:
    '''Hyper parameters'''
    num_exp=0
    # pipeline
    prepro = False  # if True, run `python prepro.py` first before running `python train.py`.

    #vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding E: End of Sentence

    # data
    data = "kss"
    test_data = "ko.txt"

    max_duration = 10.0

    # signal processing
    sr = 22050 # Sample rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds (overlab length : 0.0125 -> 0.025) #0.025로 줄이려면 다른것도 손봐야하는듯(error)
    frame_length = 0.05 # seconds (frame length : 0.05 -> 0.1) ##0.1로 줄이려면 다른것도 손봐야하는듯(error)
    hop_length = int(sr*frame_shift) # samples.
    win_length = int(sr*frame_length) # samples.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.5 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations (griffin lim iter num : 50->100)
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20
    fmin = 125  #Set this to 75 if your speaker is male! if female, 125 should help taking off noise. (To test depending on dataset)
    fmax = 7600

    # model
    embed_size = 128 # alias = E (256->128로 바꿨음) / shape이 다르면 checkpoint 사용 불가
    encoder_num_banks = 5 #16 -> 5로 바꿈 (training data의 양이 적을수록 K는 작아야 함(모델링하는 n-gram의 최대 n이 작아야 한다))
    #DB양 9~11시간 : 5 / 6시간 : 3
    decoder_num_banks = 3 #8->3
    num_highwaynet_blocks = 2 #4->2
    r = 5 # Reduction factor. Paper => 2, 3, 5
    #Decoder time step 당 하나가 아닌 여러 frame의 spectrogram을 예상함으로써 훈련시간, 합성시간, 모델 size를 줄임
    #연속한 frame의 spectrogram끼리 서로 겹치는 정보가 많기 때문에 가능
    #디코더 time step당 예측하는 frame의 갯수를 reduction factor(r)이라 부름

    dropout_rate = .5

    if num_exp == 0:
        vocab = [u"␀", u"␃", " ", "!", ",", ".", "?", 'aa', 'c0', 'cc', 'ch', 'ee', 'h0', 'ii', 'k0', 'kf', 'kh', 'kk', 'ks', 'lb', 'lh', 'lk', 'll', 'lm', 'lp',
         'ls', 'lt', 'mf', 'mm', 'nc', 'nf', 'nh', 'nn', 'ng', 'oh', 'oo', 'p0', 'pf', 'ph', 'pp', 'ps', 'qq', 'rr', 's0',
         'ss', 't0', 'tf', 'th', 'tt', 'uu', 'vv', 'wa', 'we', 'wi', 'wo', 'wq', 'wv', 'xi', 'xx', 'ya', 'ye', 'yo',
         'yq', 'yu', 'yv']
    elif num_exp == 1:
        vocab = u'''␀␃ !,.?ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ'''
    elif num_exp == 2:
        vocab = u'''␀␃ !,.?ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄾㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ''' # HCJ
    elif num_exp == 3:
        vocab = u'''␀␃ !,.?ᄀᄂᄃᄅᄆᄇᄉᄋᄌᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆫᆮᆯᆷᆸᆺᆼᆽᆾᆿᇀᇁᇂ'''
    elif num_exp == 4:
        vocab = u'''␀␃ !,.?ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ''' # HCJ. single consonants only.
    max_N, max_T = 123, 162
    #max_N : data_load - synthesize(합성할 텍스트 길이에 따라)
    #max_T : 안쓰이는듯


    # training scheme
    lr = 0.002 # Initial learning rate. (0.001 -> 0.002)
    logdir = "logdir/{}".format(num_exp)
    sampledir = 'samples/{}'.format(num_exp)
    batch_size = 16 #32 -> 16
    num_iterations = 40000



