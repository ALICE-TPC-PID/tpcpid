from sklearn import preprocessing
import torch.nn as nn
import torch.optim as optim

########### Import the Neural Network class ###########

DICT_MEAN = {
    "DATA_SPLIT": {
        "shuffle": True,
        "test_size": 0.1,
    },
    "DATA_LOADER": {
        "batch_sizes": [262144,32768,2048,512],
        "X_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "y_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "transform_data": False,
        "shuffle_every_epoch": True,
        "copy_to_device": False
    },
    "NET_DEF": {
        "n_neurons_intermediate": 12,
        "n_layers": 10,
        "n_neurons_output": 1
    },
    "NET_SETTINGS": {
        "w_init": nn.init.xavier_normal_,
        "scale_data": False,
        "gain": 2.5,
        "verbose": True
    },
    "NET_TRAINING": {
        "epochs_ls": [0,30,50,80],
        "weights": False,
        "optimizer": optim.Adam,
        "weight_decay": 0,
        "scheduler": optim.lr_scheduler.ReduceLROnPlateau,
        "learning_rate": 0.001,
        "copy_to_device": 1,
        "patience": 10,
        "factor": 0.5,
        "verbose": True
    }
}

DICT_SIGMA = {
    "DATA_SPLIT": {
        "shuffle": True,
        "test_size": 0.1,
    },
    "DATA_LOADER": {
        "batch_sizes": [262144,32768,2048,512],
        "X_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "y_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "transform_data": False,
        "shuffle_every_epoch": True,
        "copy_to_device": False
    },
    "NET_DEF": {
        "n_neurons_intermediate": 12,
        "n_layers": 10,
        "n_neurons_output": 1
    },
    "NET_SETTINGS": {
        "w_init": nn.init.xavier_normal_,
        "scale_data": False,
        "gain": 2.5,
        "verbose": True
    },
    "NET_TRAINING": {
        "epochs_ls": [0,30,50,80],
        "weights": False,
        "optimizer": optim.Adam,
        "weight_decay": 0,
        "scheduler": optim.lr_scheduler.ReduceLROnPlateau,
        "learning_rate": 0.001,
        "copy_to_device": 1,
        "patience": 10,
        "factor": 0.5,
        "verbose": True
    }
}

DICT_FULL = {
    "DATA_SPLIT": {
        "shuffle": True,
        "test_size": 0.1,
    },
    "DATA_LOADER": {
        "batch_sizes": [262144,32768,2048,512],
        "X_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "y_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "transform_data": False,
        "shuffle_every_epoch": True,
        "copy_to_device": False
    },
    "NET_DEF": {
        "n_neurons_intermediate": 8,
        "n_layers": 6,
        "n_neurons_output": 2
    },
    "NET_SETTINGS": {
        "w_init": nn.init.xavier_normal_,
        "scale_data": False,
        "gain": 5./3.,
        "verbose": True
    },
    "NET_TRAINING": {
        "epochs_ls": [0,30,50,80],
        "weights": False,
        "optimizer": optim.Adam,
        "weight_decay": 0,
        "scheduler": optim.lr_scheduler.ReduceLROnPlateau,
        "learning_rate": 0.001,
        "copy_to_device": 1,
        "patience": 10,
        "factor": 0.5,
        "verbose": True
    }
}


class model(nn.Module):
    
    def __init__(self, dict_config):
        super(model, self).__init__()
        self.net_def = dict_config["NET_DEF"]
    
        self.network = nn.Sequential(
            nn.Linear(self.net_def["n_neurons_input"], self.net_def["n_neurons_intermediate"]),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(self.net_def["n_neurons_intermediate"], self.net_def["n_neurons_intermediate"]), nn.ReLU()) for i in range(self.net_def["n_layers"]-1)],
            nn.Linear(self.net_def["n_neurons_intermediate"], self.net_def["n_neurons_output"])
        )
        
    def forward(self, x):
        return self.network(x)
