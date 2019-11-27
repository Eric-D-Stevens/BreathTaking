import torch
from torch import nn

class AutoEncoder(nn.Module):

    def __init__(self, d_in, relu1, sigmoid1, relu2, sigmoid2, encoding):
        super(AutoEncoder, self).__init__()

        # encoder block
        self.encoder = nn.Sequential(
            nn.Linear(d_in, relu1),
            nn.ReLU(True),
            nn.Linear(relu1, sigmoid1),
            nn.Sigmoid(),
            nn.Linear(sigmoid1, relu2),
            nn.ReLU(True),
            nn.Linear(relu2, sigmoid2),
            nn.Sigmoid(),
            nn.Linear(sigmoid2, encoding))

        # decoder block
        self.decoder = nn.Sequential(
            nn.Linear(encoding, relu1),
            nn.ReLU(True),
            nn.Linear(relu1, sigmoid1),
            nn.Sigmoid(),
            nn.Linear(sigmoid1, relu2),
            nn.ReLU(True),
            nn.Linear(relu2, sigmoid2),
            nn.Sigmoid(),
            nn.Linear(sigmoid2, d_in))

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SecondOrderAutoEncoder(nn.Module):

    def __init__(self, prior_model, d_in, relu1, sigmoid1, relu2, sigmoid2, encoding):
        super(SecondOrderAutoEncoder, self).__init__()

        self.prior_model = prior_model
        self.prior_model.eval()



        # input size to encoder post model 1
        d_in_encoder = int(d_in/10)

        # encoder block
        self.encoder = nn.Sequential(
            nn.Linear(d_in_encoder, relu1),
            nn.ReLU(True),
            nn.Linear(relu1, sigmoid1),
            nn.Sigmoid(),
            nn.Linear(sigmoid1, relu2),
            nn.ReLU(True),
            nn.Linear(relu2, sigmoid2),
            nn.Sigmoid(),
            nn.Linear(sigmoid2, encoding))

        # decoder block
        self.decoder = nn.Sequential(
            nn.Linear(encoding, relu1),
            nn.ReLU(True),
            nn.Linear(relu1, sigmoid1),
            nn.Sigmoid(),
            nn.Linear(sigmoid1, relu2),
            nn.ReLU(True),
            nn.Linear(relu2, sigmoid2),
            nn.Sigmoid(),
            nn.Linear(sigmoid2, 320),
            nn.ReLU(True),
            nn.Linear(320, 640),
            nn.ReLU(True),
            nn.Linear(640, 1280),
            nn.ReLU(True), nn.Linear(1280, 2400), nn.Sigmoid(), nn.Linear(2400, 3200))

    def chunk_input(self, x):

        chunk = int(x.shape[1] / 10)
        mini_chunk = int(chunk / 10)
        xs = torch.zeros((x.shape[0], chunk)).to(x.device)

        for i in range(10):
            in_chunk = x[:, i * chunk:(i + 1) * chunk]
            xs[:, i * mini_chunk:(i + 1) * mini_chunk] = self.prior_model.encoder(in_chunk)
        return xs


    def forward(self, x):

        if x.shape[0] == 3200:
            x = torch.unsqueeze(x, 0)

        xs = self.chunk_input(x)
        xs = self.encoder(xs)
        xs = self.decoder(xs)
        return xs

    def encode_3200_32(self, x):
        self.eval()
        xs = self.chunk_input(x)
        xs = self.encoder(xs)
        return xs


class ThirdOrderAutoEncoder(nn.Module):

    def __init__(self, prior_model, d_in, relu1, sigmoid1, relu2, sigmoid2, encoding):
        super(ThirdOrderAutoEncoder, self).__init__()

        self.prior_model = prior_model
        self.prior_model.eval()

        # input size to encoder post model 1
        d_in_encoder = int(d_in/10)

        # encoder block
        self.encoder = nn.Sequential(
            nn.Linear(320, relu1),
            nn.ReLU(True),
            nn.Linear(relu1, sigmoid1),
            nn.Sigmoid(),
            nn.Linear(sigmoid1, relu2),
            nn.ReLU(True),
            nn.Linear(relu2, sigmoid2),
            nn.Sigmoid(),
            nn.Linear(sigmoid2, encoding))

        # decoder block
        self.decoder = nn.Sequential(
            nn.Linear(encoding, relu1),
            nn.ReLU(True),
            nn.Linear(relu1, sigmoid1),
            nn.Sigmoid(),
            nn.Linear(sigmoid1, relu2),
            nn.ReLU(True),
            nn.Linear(relu2, sigmoid2),
            nn.Sigmoid(),
            nn.Linear(sigmoid2, 320),
            nn.Sigmoid(),
            nn.Linear(320, 640),
            nn.ReLU(True),
            nn.Linear(640, 1280),
            nn.ReLU(True),
            nn.Linear(1280, 2800),
            nn.ReLU(True),
            nn.Linear(2800, 32000))


    def forward(self, x):

        if x.shape[0] == 32000:
            x = torch.unsqueeze(x, 0)

        chunk = int(x.shape[1]/10)
        mini_chunk = int(chunk/100)
        xs = torch.zeros((x.shape[0], int(chunk/10))).to(x.device)

        for i in range(10):
            in_chunk = x[:, i*chunk:(i+1)*chunk]
            xs[:, i*mini_chunk:(i+1)*mini_chunk] = self.prior_model.encode_3200_32(in_chunk)

        xs = self.encoder(xs)
        xs *= 100 # amp
        xs = self.decoder(xs)
        return xs






def SpectralConverfenceLoss(output, target):
    loss = torch.sqrt(torch.sum((target - output) ** 2) / torch.sum(target ** 2))
    return loss




