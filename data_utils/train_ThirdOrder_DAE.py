import os
import torch
import utils
from auto_encoder.data_loader import OneSecondDataset, TransformFFT
from torch.utils.data import DataLoader
from auto_encoder.models import AutoEncoder, SecondOrderAutoEncoder, ThirdOrderAutoEncoder
from data_utils.DatsetAE import AutoEncoderDatatSet

def train(prior_model, sample_seconds, num_samples):
    # CUDA for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    torch.cuda.empty_cache()
    print("running on ", device)

    dataset = AutoEncoderDatatSet(utils.long_data, sample_size_seconds=sample_seconds, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    d_in = dataset[1].shape[0]
    print(d_in)

    # MODEL LOSS OPTIMIZER
    AE = ThirdOrderAutoEncoder(prior_model, d_in, relu1=140,
                                sigmoid1=100, relu2=60,
                                sigmoid2=40,encoding=32).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(AE.parameters(), lr=0.001)




    epochs = 100
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

    torch.save(AE, os.path.join(utils.root, 'trained_modles', 's100k_t1s_b300_e100.model'))
    return AE


prior_model = torch.load(os.path.join(utils.root, 'trained_modles', 's100k_t100ms_b300_e1000.model'))
train(prior_model, 1, 10000)
