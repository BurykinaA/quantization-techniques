from torch import nn
from ADC.quantized_layers import LinearADC, LinearQuant


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256), 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x) 


class MLPADC(nn.Module):
    def __init__(self, bx=8, bw=8, ba=8, k=4):
        super(MLPADC, self).__init__()
        self.layers = nn.Sequential(
            LinearADC(784, 256, bx, bw, ba, k),
            nn.ReLU(),
            LinearADC(256, 128, bx, bw, ba, k),
            nn.ReLU(),
            LinearADC(128, 10, bx, bw, ba, k)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.layers(x)
    

class MLPQuant(nn.Module):
    def __init__(self, bx=8, bw=8):
        super(MLPQuant, self).__init__()
        self.layers = nn.Sequential(
            LinearQuant(784, 256, bx, bw),
            nn.ReLU(),
            LinearQuant(256, 128, bx, bw),
            nn.ReLU(),
            LinearQuant(128, 10, bx, bw)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)
