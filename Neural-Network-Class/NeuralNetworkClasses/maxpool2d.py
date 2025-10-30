import torch.nn as nn

class maxpool2d_layer(nn.Module):
    
    def __init__(self, params, activation, weight_init, verbose=False, **options):
    
        super().__init__()

        self.kernel_size, self.stride = params[0], params[1]
        self.act = activation
        self.maxpool = nn.MaxPool2d(self.kernel_size, self.stride, **options)
        
        if verbose:
            print("MaxPool2D, Kernel size: ", self.kernel_size, ", stride: ", self.stride)
            
    def forward(self, X):
        
        return self.act(self.maxpool(X))
