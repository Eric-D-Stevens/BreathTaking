from scipy.fftpack import fft, ifft
from scipy.io import wavfile
import os
import numpy as np

def generate_mag_phase_array_from_directory(directory):

    fft_div = 4
    out_array = 'x'

    files = os.listdir(directory)
    for ind, file in enumerate(files):
        fpath = os.path.join(directory, file)
        print(fpath)
        f, data = wavfile.read(fpath)

        if out_array == 'x':
            do_len = int(len(data)/fft_div)
            out_array = np.zeros((len(files), 2, do_len), dtype=data.dtype)

        fourier = fft(data, n=do_len)
        out_array[ind, 0] = np.real(fourier)
        out_array[ind, 1] = np.imag(fourier)

    return out_array

