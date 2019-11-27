from scipy.io import wavfile
from random import randint
import os

def generate_slice(input_filename,
                   output_location,
                   fout_prefix = 'out',
                   max_lenght_s = 10,
                   min_lenght_s = 3,
                   num_clips=1000):

    # load a large audio clip
    fs, data = wavfile.read(input_filename)
    #data = data[:,0] # use only one channel

    max_samp, min_samp = int(max_lenght_s*fs), int(min_lenght_s*fs)
    data_lenght = len(data)

    for i in range(num_clips):
        start = randint(0, data_lenght-max_samp)
        end = start + randint(min_samp, max_samp)

        clip = data[start:end]
        out_filename = os.path.join(output_location, fout_prefix+str(i)+'.wav')
        print('Writing: ', out_filename)
        wavfile.write(out_filename, fs, clip)








