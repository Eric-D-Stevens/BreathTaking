from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
from scipy.io import wavfile
from scipy.fftpack import fft
import numpy as np

class OneSecondDataset(Dataset):
    '''   '''

    def __init__(self, one_dir, transform=None):
        self.one_dir = one_dir
        self.file_list = os.listdir(self.one_dir)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_rate, data = wavfile.read(os.path.join(self.one_dir, self.file_list[idx]))
        sample = {'data': data, 'sample_rate': sample_rate}

        if self.transform:
            sample = self.transform(sample)

        return sample

class TransformFFT(object):
    ''' '''

    def __call__(self, sample):
        wav = fft(sample['data'])
        wav_r = np.real(wav)
        wav_i = np.imag(wav)
        sample['data'] = torch.tensor(np.concatenate((wav_r, wav_i)))

        return sample


