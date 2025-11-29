import torch.nn as nn

class dropout_layer(nn.Module):
    
    def __init__(self, params, activation, weight_init, verbose=False, **options):

        super().__init__()

        self.rate, self.act = params, activation
        self.drop = nn.Dropout(self.rate)
        
        if verbose:
            print("Dropout layer, dropout-rate: ", self.rate, ", activation: ", self.act)
        

    def forward(self, X):
        
        return self.act(self.drop(X))