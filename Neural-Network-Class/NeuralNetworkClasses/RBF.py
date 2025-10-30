import torch.nn as nn
import torch

class gaussian_layer(nn.Module):
    
    def __init__(self, params, activation, weight_init, verbose=False, **options):

        super().__init__()

        self.size_in, self.size_out, self.act = params[0], params[1], activation
        self.fc = nn.Linear(self.size_in, self.size_out)
        self.centers = nn.Parameter(torch.Tensor(self.size_out, self.size_in))
        self.log_sigmas = nn.Parameter(torch.Tensor(self.size_out))
        self.reset_parameters()
        
        if verbose:
            print("RBF-gaussian, Input size: ", self.size_in, ", Output size: ", self.size_out, ", activation: ", self.act)
        
    def reset_parameters(self):
        nn.init.normal_(self.centers, 0., 1.)
        nn.init.constant_(self.log_sigmas, 0.)
        
    def gaussian(self, alpha):
        phi = torch.exp(-1.*alpha.pow(2.))
        return phi

    def forward(self, X):
        size = (X.size(0), self.size_out, self.size_in)
        x = X.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        exponent = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.gaussian(exponent)