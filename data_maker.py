# Input과 Target data를 만든다.
# Input data는 spectrogram
# target data: 위상차

import os
from sklearn.externals import joblib

import numpy as np
import scipy.io as sio
import scipy.io.wavfile

from librosa.core import stft
import soundfile as sf

def get_stft(signal):

    n_fft = 512
    win_len = 512
    hop_len = 128
    st = stft(signal,
              n_fft = n_fft,
              win_length = win_len,
              hop_length = hop_len)

    return st


if __name__ == "__main__":

    # wave 파일 위치
    wavFilePath = 'D:/work/GrandChallenge/6월14일2ch데이터녹음/190614_1623_ch1[30]_ch2[30].wav'

    # rate, data = sio.wavfile.read(wavFilePath)

    data, rate = sf.read(wavFilePath)

    # m1 = data[0, :32000]
    # m2 = data[1, :32000]
    # m1 = data[:32000, 0]
    # m2 = data[:32000, 1]
    m1 = data[:31999, 0]
    m2 = data[:31999, 1]

    # m1 = data[:32000, 0]
    # m2 = data[:32000, 1]

    m1_tf = get_stft(m1)
    m2_tf = get_stft(m2)


    b = 0
    #
    # stft(signal,
    #      n_fft=self.n_fft,
    #      win_length=self.win_len,
    #      hop_length=self.hop_len)
    #
    #
    # ## wav file를 spectrogram
    # for i, source_info in enumerate(mixture_info['sources_ids']):
    #     wav, fs = self.load_wav(source_info)
    #     if self.normalize_audio_by_std:
    #         wav = wav / np.std(wav)
    #     mixture_info['sources_ids'][i]['fs'] = int(fs)
    #     mixture_info['sources_ids'][i]['wav'] = wav
    #
    # tf_representations = self.get_tf_representations(mixture_info)
    #
    # return tf_representations


    ## GT 만들기


    a = 0