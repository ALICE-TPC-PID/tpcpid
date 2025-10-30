import torch.nn as nn

class fc_layer(nn.Module):
    
    def __init__(self, params, activation, weight_init, verbose=False, **options):

        super().__init__()

        self.size_in, self.size_out, self.act = params[0], params[1], activation
        self.fc = nn.Linear(self.size_in, self.size_out)
        
        if verbose:
            print("Fully-connected, Input size: ", self.size_in, ", Output size: ", self.size_out, ", activation: ", self.act)

        ######## Initialize the weights #############
        weight_init(self.fc.weight, **options)
        

    def forward(self, X):
        
        return self.act(self.fc(X))