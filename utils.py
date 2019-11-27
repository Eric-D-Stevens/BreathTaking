import os
from scipy.fftpack import fft, ifft
import numpy as np
import sounddevice as sd
import torch
import matplotlib.pyplot as plt


root = os.path.dirname(os.path.abspath(__file__))
one_dir = os.path.join(root, 'data', 'one_second')
ten_ms = os.path.join(root, 'data', 'ten_ms')
long_data = os.path.join(root, 'data', 'long_audio', 'hd1_16k.wav')


def play_flattened_fft(d, fs=16000):

    d = d.detach().numpy()
    mid = int(len(d)/2)
    r = d[:mid]
    i = d[mid:]

    x = np.real(ifft(r+1j*i))
    x_tend = np.repeat(x, 100)
    sd.play(x_tend, fs)



def get_time_domain(d):

    if type(d) == torch.Tensor:
        print('getTD: type is tensor')
        d = d.detach().numpy()
        d = np.atleast_2d(d)
        d = d[0]

    half = int(len(d)/2)
    wav = ifft(d[:half] + 1.0j*d[half:])
    return np.real(wav)

def play_sound(d, sr=16000):

    d = d.detach().numpy()
    d = np.atleast_2d(d)
    d = d[0]

    wav = get_time_domain(d)
    print('wav shape: ', wav.shape, wav.dtype)
    sd.play(wav, sr)
    return wav

def plot_io(model, dataset, idx):

    plt.clf()

    xi = torch.unsqueeze(dataset[idx], 0)
    model.eval()

    xo = model(xi)
    xo = xo.detach().numpy()
    xo = xo[0]

    xi = xi.detach().numpy()
    xi = xi[0]



    plt.plot(xi, label='input', alpha=.5)
    plt.plot(xo, label='output', alpha=.5)
    plt.legend()


def get_wavs(model, t_in, wav, sr=16000):

    model.eval()

    wav_chunks = np.reshape(wav, (int(len(wav)/(t_in*sr)), int(t_in*sr)))
    out_chunks = np.zeros((wav_chunks.shape[0], int(2*t_in*sr)))
    wav_out = np.zeros(wav_chunks.shape)

    for i in range(wav_chunks.shape[0]):
        ft = fft(wav_chunks[i])
        if t_in == 0.01:
            fri = torch.Tensor(np.concatenate((np.real(ft), np.imag(ft)), 0))
        else:
            fri = torch.unsqueeze(torch.Tensor(np.concatenate((np.real(ft), np.imag(ft)), 0)),0)
        out_chunks[i] = model(fri).detach().numpy()
        wav_out[i] = get_time_domain(out_chunks[i])
        print(wav_out.shape)

    wav_out = np.reshape(wav_out, len(wav))

    return (wav, wav_out)







