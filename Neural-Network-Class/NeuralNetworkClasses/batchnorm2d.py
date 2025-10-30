import torch.nn as nn

class batchnorm2d_layer(nn.Module):
    
    def __init__(self, params, activation, weight_init, verbose=False, **options):
        
        super().__init__()
        
        self.channels = params
        self.act = activation
        self.bn2d = nn.BatchNorm2d(self.channels)
        
        print("Batchnorm, Input size = Output size, Channels: ", self.channels, ", activation: ", self.act)
        
    def forward(self, X):
        
        return self.act(self.bn2d(X))
    