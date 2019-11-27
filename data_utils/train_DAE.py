import os
import torch
import utils
from auto_encoder.data_loader import OneSecondDataset, TransformFFT
from torch.utils.data import DataLoader
from auto_encoder.models import AutoEncoder, SpectralConverfenceLoss
from data_utils.DatsetAE import AutoEncoderDatatSet

def train():
    # CUDA for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    torch.cuda.empty_cache()
    print("running on ", device)

    dataset = AutoEncoderDatatSet(utils.long_data, sample_size_seconds=0.01, num_samples=100000)
    dataloader = DataLoader(dataset, batch_size=300, shuffle=True)

    d_in = dataset[1].shape[0]
    print(d_in)

    # MODEL LOSS OPTIMIZER
    AE = AutoEncoder(d_in,
                     relu1=140,
                     sigmoid1=100,
                     relu2=60,
                     sigmoid2=40,
                     encoding=32).to(device)

    criterion = torch.nn.MSELoss()
    #criterion = SpectralConverfenceLoss
    optimizer = torch.optim.Adam(AE.parameters(), lr=0.0001)




    epochs = 1000
    for e in range(epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            AE.train()
            optimizer.zero_grad()

            # print(i_batch, sample_batched['data'].size())
            sample_batched = sample_batched.to(device)

            # a Tensor of output data.
            y_ = AE(sample_batched)
            y = sample_batched

            loss = criterion(y_, y)

            loss.backward(loss)
            optimizer.step()

        print("Epoch: ", e, "Loss: ", loss)

    torch.save(AE, os.path.join(utils.root, 'trained_modles', 'trained_.model'))
    return AE


train()
