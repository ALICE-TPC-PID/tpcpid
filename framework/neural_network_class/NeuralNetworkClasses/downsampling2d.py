import torch.nn as nn
from conv2d import conv2d_layer
from batchnorm2d import batchnorm2d_layer

class downsampling2d_channels_layer(nn.Module):
    
    def __init__(self, params, activation, weight_init, verbose=False, **options):
        
        super().__init__()
        
        self.params = params
        self.weight_init = weight_init
        
        if verbose:
            print("Downsampling (Conv2d and BatchNorm), Input size: ", params[0], ", Output size: ", params[1])
        
        self.conv1 = nn.Conv2d(self.params[0], self.params[1], kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.params[1])
            
    def forward(self, X):
        
        out = self.conv1(X)
        out = self.bn1(out)
        
        return out