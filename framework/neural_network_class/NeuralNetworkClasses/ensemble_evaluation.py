import torch
import onnxruntime as ort
import numpy as np
from tqdm import tqdm

class ensemble:
    
    def __init__(self, list_model_files, mode="ONNX"):
        
        super(ensemble, self).__init__()
        
        self.network_files = list_model_files
        self.networks = []
        self.mode = mode
        
        for entry in tqdm(self.network_files):
            if mode == "TORCH":
                self.networks.append(torch.load(entry))
            if mode == "ONNX":
                ort_sess = ort.InferenceSession(entry)
                self.networks.append(ort_sess)

        print("Networks loaded!")
        
    def eval(self, data, reshape = False):
        
        values = []
        
        for net in self.networks:
            if reshape:
                if self.mode == "TORCH":
                    values.append(net(data).detach().numpy().flatten().reshape(reshape))
                elif self.mode == "ONNX":
                    values.append(net.run(None, {'input': (torch.tensor(data).float()).numpy()}).reshape(reshape))
            else:
                if self.mode == "TORCH":
                    values.append(net(data).detach().numpy().flatten())
                elif self.mode == "ONNX":
                    values.append(net.run(None, {'input': (torch.tensor(data).float()).numpy()})[0].flatten())
                    
        return values
        
    def mean(self, data, axis=0, reshape=False):
        
        values = self.eval(data, reshape)
        
        return np.mean(values, axis=axis)
        
    def sigma(self, data, axis=0, reshape = False):
        
        values = self.eval(data, reshape)
        
        return np.std(values, axis=axis)
        