import os
import torch
import utils
from auto_encoder.data_loader import OneSecondDataset, TransformFFT
from torch.utils.data import DataLoader
from auto_encoder.models import AutoEncoder
from data_utils.DatsetAE import AutoEncoderDatatSet

def train1():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    #use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()
    print("running on ", device)

    fft_trans = TransformFFT()
    dataset = OneSecondDataset(one_dir=utils.ten_ms, transform=fft_trans)

    dataset = AutoEncoderDatatSet(utils.long_data, sample_size_seconds=0.01, num_samples=10000, transform='mp')
    dataloader = DataLoader(dataset, batch_size=500, shuffle=True)

    d_in = dataset[1]['data'].shape[0]
    print(d_in)

    # MODEL LOSS OPTIMIZER
    AE = AutoEncoder(d_in,
                     relu1=160,
                     sigmoid1=80,
                     encoding=20,
                     relu2=80,
                     tanh1=160).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(AE.parameters(), lr=0.0001)




    epochs = 200
    for e in range(epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            AE.train()
            optimizer.zero_grad()

            #print(i_batch, sample_batched['data'].size())
            sample_batched = sample_batched['data']
            sample_batched = sample_batched.to(device)

           # a Tensor of output data.
            y_= AE(sample_batched)
            y = sample_batched

            loss = criterion(y_, y)

            loss.backward(loss)
            optimizer.step()

            print("Epoch: ", e, "Loss: ", loss)

    torch.save(AE, os.path.join(utils.root, 'trained_modles', 'trained_10ms_20nodes.model'))
    return AE
