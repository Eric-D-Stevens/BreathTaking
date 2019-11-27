import utils
from data_utils.DatsetAE import AutoEncoderDatatSet
import torch
import os
import sounddevice as sd
sr = 16000

ds = AutoEncoderDatatSet(utils.long_data, sample_size_seconds=5, num_samples=10)

A = torch.load(os.path.join(utils.root,  'trained_modles', 's100k_t1s_b300_e100.model'))
A = A.to('cpu')
A.eval()

w = utils.get_time_domain(ds[7])

i, o = utils.get_wavs(A, 1, w)

