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
    data = "/home/jeon/Downloads/kss"
    test_data = "ko.txt"

    max_duration = 10.0

    # signal processing
    sr = 22050 # Sample rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples.
    win_length = int(sr*frame_length) # samples.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.5 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # model
    embed_size = 128 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
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



    # training scheme
    lr = 0.001 # Initial learning rate.
    logdir = "logdir/{}".format(num_exp)
    sampledir = 'samples/{}'.format(num_exp)
    batch_size = 16
    num_iterations = 400000



