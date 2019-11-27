import utils
from data_utils.DatsetAE import AutoEncoderDatatSet
import torch
import os

ds = AutoEncoderDatatSet(utils.long_data, sample_size_seconds=.1, num_samples=100)

A = torch.load(os.path.join(utils.root,  'trained_modles', 's100k_t100ms_b300_e1000.model'))
A = A.to('cpu')
A.eval()

d = torch.Tensor(ds[3])
A(d)
