import torch.nn as nn

class conv1d_layer(nn.Module):
    
    def __init__(self, params, activation, weight_init, verbose=False, **options):

        super().__init__()

        try: 
            self.in_ch, self.out_ch, self.kernel_size, self.stride, self.act = params[0], params[1], params[2], params[3], activation
        except:
            self.in_ch, self.out_ch, self.kernel_size, self.act = params[0], params[1], 3, 1, activation
            print("The neuron was falsly initialized: params[2] does not exist! A kernel_size of 3x3 will be used.")
            
        self.conv = nn.Conv1d(self.in_ch, self.out_ch, self.kernel_size, self.stride, **options)
        
        ######## Initialize the weights #############
        weight_init(self.conv.weight)
        
        if verbose:
            print("Conv1D, Input channels: ", self.in_ch, ", Output channels: ", self.out_ch, ", Kernel size: ", self.kernel_size, ", Stride: ", self.stride, ", activation: ", self.act)

    def forward(self, X):

        return self.act(self.conv(X))