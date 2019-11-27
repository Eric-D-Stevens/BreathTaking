from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import numpy as np
from random import randint
import sounddevice as sd

class AutoEncoderDatatSet(Dataset):

    def __init__(self, long_filename, sample_size_seconds=1.0, num_samples=10, transform='ri'):

        # store needed inputs
        self.num_samples = num_samples
        self.sample_size_seconds = sample_size_seconds
        self.transform = transform

        # load the raw wav and store the sample rate
        self.sample_rate, raw_wav = wavfile.read(long_filename)
        raw_wav = raw_wav.astype(np.float32)

        # create the data set
        self.wav_length = np.int(np.int(self.sample_rate*self.sample_size_seconds))

        # randomly select samples
        max_start = len(raw_wav)-self.wav_length
        self.data = np.zeros((self.num_samples, 2*self.wav_length))

        for i in range(num_samples):
            start = randint(0, max_start)
            ft = fft(raw_wav[start:start+self.wav_length])

            # concat real/img
            if transform == 'ri':
                self.data[i] = np.concatenate((np.real(ft), np.imag(ft))).astype(np.float32)

            # concat mag/phase
            elif transform == 'mp':
                self.data[i] = np.concatenate((np.absolute(ft), np.angle(ft))).astype(np.float32)


            else:
                raise ValueError("unknown transform type")

        print("Data Loaded")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        d = self.data[idx]
        d = d.astype(np.float32)
        return torch.Tensor(d)

    def get_time_domain(self, idx):
        d = self.data[idx]
        half = int(len(d)/2)

        if self.transform == 'ri':
            wav = ifft(d[:half] + 1.0j*d[half:])
        elif self.transform == 'mp':
            wav = ifft(d[:half]*np.cos(d[half:]) + 1.0j*d[:half]*np.sin(d[half:]))
        else:
            raise ValueError('unrecognized transform')

        return np.real(wav)

    def play_sound(self, idx):
        wav = self.get_time_domain(idx)
        print('wav shape: ', wav.shape, wav.dtype)
        sd.play(wav, self.sample_rate)
        print(wav)
