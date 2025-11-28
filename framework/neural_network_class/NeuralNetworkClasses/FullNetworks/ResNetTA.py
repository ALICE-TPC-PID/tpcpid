import torch
import torch.nn as nn


class ConvBlockEncoder(nn.Module):
    
    def __init__(self, size_in, kernel_size=(1,1), padding=0, dropout_rate=0.3, pool_filter=2):
        
        super().__init__()
        
        self.conv = nn.Conv2d(size_in,size_in, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout_rate)
        self.maxpool = nn.MaxPool2d(pool_filter)

    def forward(self, X):

        out = self.conv(X)
        out = out + X
        out = self.dropout(out)
        out = self.maxpool(out)
        
        return out
    
class LinearBlock(nn.Module):
    
    def __init__(self, size_in, size_out):
        
        super().__init__()
        
        self.linear1 = nn.Linear(size_in, size_in)
        self.linear2 = nn.Linear(size_in, size_in)
        self.linear3 = nn.Linear(size_in, size_out)
        self.relu = nn.ReLU
        
    def forward(self, X):
        
        out = self.relu(self.linear3(self.relu(self.linear2(self.relu(self.linear1(X))))))
        
        return out


class ResNetTA(nn.Module):

    def __init__(self, size_in, n_features, conv_blocks=3):
        super().__init__()

        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Conv2d(1,size_in,kernel_size=(1,1)))

        self.layers.append(ConvBlockEncoder(size_in, kernel_size=(1,1), padding=0))

        for i in range(conv_blocks):
            self.layers.append(ConvBlockEncoder(size_in, kernel_size=(3,1), padding=3-1))
         
        self.layers.append(LinearBlock(size_in*3*3, 1))

    def forward(self, X):

        out = X
        for layer in self.layers:
            out = layer.forward(out)

        return out
