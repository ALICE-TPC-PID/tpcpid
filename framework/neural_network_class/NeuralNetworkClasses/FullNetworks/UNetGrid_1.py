import torch
import torch.nn as nn
import numpy as np


class UNetGrid_1(nn.Module):
    
    def __init__(self, size_in, size_out=1):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        
        self.upsample1 = nn.Linear(size_in, 3*size_in)
        self.bn1 = nn.BatchNorm1d(3*size_in)
        self.conv1 = nn.Conv1d(1, 3, 5, stride = 2)
        self.linear1 = nn.Linear(3*12, 12)
        self.linear2 = nn.Linear(12,size_in)

        self.upsample2 = nn.Linear(size_in,12)
        self.upsample3 = nn.Linear(12,3*12)
        self.conv2 = nn.Conv1d(1,3,5,stride=2)
        self.bn2 = nn.BatchNorm1d(3*16)
        self.downsample1 = nn.Linear(3*16,1)
        
    def forward(self, X):
        out = X
        out = self.upsample1(out)
        out = self.bn1(out)
        out = torch.reshape(out, [out.size()[0],1,out.size()[1]])
        out = self.conv1(out)
        out = torch.reshape(out, [out.size()[0],out.size()[1]*12])
        out = self.linear1(out)
        out = self.linear2(out)

        out = self.upsample2(out)
        out = self.upsample3(out)
        out = torch.reshape(out, [out.size()[0],1,out.size()[1]])
        out = self.conv2(out)
        out = torch.reshape(out, [out.size()[0],out.size()[1]*16])
        out = self.bn2(out)
        out = self.downsample1(out)

        return out