import torch.nn as nn

class resnet_basic_layer(nn.Module):
    
    def __init__(self, params, activation, weight_init, verbose=False, **options):
        
        super().__init__()
        
        self.params = params
            
        self.act = activation
        self.weight_init = weight_init
        
        '''
        if verbose:
            
            print("ResNet layer:\n")
            print("\tConv2D, Input channels: ", self.params[0][0], 
                  ", Output channels: ", self.params[0][1], 
                  ", Kernel size: ", self.params[0][2],
                  ", activation: ", nn.Identity())
            print("\tBatchnorm, Input size = Output size, Channels: ", 
                  self.params[0][1], ", activation: ", self.act[0])
            print("\tConv2D, Input channels: ", self.params[1][0], 
                  ", Output channels: ", self.params[1][1], 
                  ", Kernel size: ", self.params[1][2],
                  ", activation: ", nn.Identity())
            print("\tBatchnorm, Input size = Output size, Channels: ", 
                  self.params[1][1], ", activation: ", nn.Identity())
            if self.downsample:
                print("\tDownsampling (Conv2d and BatchNorm), Input size: ", self.params[2][0], 
                      ", Output size: ", self.params[2][1])
            print("\tApplying activation: ", self.act[1], ", adding input X")
            
        if self.params[0][1] != self.params[1][0]:
            print("Output channels of first conv. layer {ch1} and input channels of second conv. layer {ch2} are not the same!".format(ch1 = self.params[0][1], ch2 = self.params[1][0]))

        if self.downsample and self.params[1][1] != self.params[2][1]:
            print("Output channels of second conv. layer {ch1} and input channels of downsapling_channels layer {ch2} are not the same!".format(ch1 = self.params[0][1], ch2 = self.params[1][0]))


        self.conv1 = conv2d_layer(self.params[0], nn.Identity(), self.weight_init, verbose = False, **options)
        self.bn1 = batchnorm2d_layer(self.params[0][1], self.act[0], self.weight_init, verbose = False, **options)
        self.conv2 = conv2d_layer(self.params[1], nn.Identity(), self.weight_init, verbose=False, **options)
        self.bn2 = batchnorm2d_layer(self.params[1][1], nn.Identity(), self.weight_init, verbose=False, **options)
        
        if self.downsample:
            self.downsampling = downsampling2d_channels_layer(self.params[2], 0, self.weight_init, verbose=False, **options)
        '''
        
        if verbose:
            print("ResNet basic layer")
          
        self.conv1 = nn.Conv2d(self.params[0], self.params[1], kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.params[1])
        
        self.conv2 = nn.Conv2d(self.params[1], self.params[1], kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.params[1])
        
        self.resample = nn.Sequential(nn.Conv2d(self.params[0], self.params[1], 1, stride=1, bias=False),
                                      nn.BatchNorm2d(self.params[1]))
        
    def forward(self, X):
        
        identity = X
        
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
            
        identity = self.resample(X)
            
        out += identity
        out = self.act(out)
        
        return out