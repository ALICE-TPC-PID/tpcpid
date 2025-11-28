import torch
import torch.nn as nn
from RBF import gaussian_layer

class Block2(nn.Module):
    
    def __init__(self, size_in, size_hidden):
        
        super().__init__()
        
        self.fn1 = nn.Linear(size_in,size_hidden)
        self.fn2 = nn.Linear(size_hidden,size_hidden)
        self.fn3 = nn.Linear(size_hidden,size_hidden)
        self.fn4 = nn.Linear(size_hidden,size_hidden)
        #self.rbf1 = gaussian_layer([size_in,size_in], nn.Identity(),weight_init=torch.nn.init.xavier_uniform_)
        self.fn5 = nn.Linear(size_hidden,size_in)

    def forward(self, X):

        out = nn.Tanh()(self.fn1(X))
        out = nn.Tanh()(self.fn2(out))
        out = nn.Tanh()(self.fn3(out))
        out = nn.Tanh()(self.fn4(out))
        out = nn.Tanh()(self.fn5(out))
        out = out + X

        return out

class last_layer(nn.Module):

    def __init__(self, size):
        
        super().__init__()
        
        self.fn1 = nn.Linear(size[0],size[1])

    def forward(self, X):

        out = nn.Identity()(self.fn1(X))

        return out    


class ResNetGrid_2(nn.Module):

    def __init__(self, block, sizes):
        super().__init__()

        self.layers = nn.ModuleList()

        for counter in range(len(sizes)-1):
            self.layers.append(block(sizes[counter][0],sizes[counter][1]))

        self.layers.append(last_layer(sizes[-1]))

    def forward(self, X):

        out = X
        for layer in self.layers:
            out = layer.forward(out)

        return out
