from sklearn import preprocessing
import itertools
import torch.nn as nn
import torch.optim as optim
import sys
import os
import json

def network_def(n_neurons_input, n_neurons_intermediate, n_neurons_output, n_layers):

    h_sizes = list(itertools.chain(*[[[n_neurons_input,n_neurons_intermediate]],[[n_neurons_intermediate,n_neurons_intermediate]]*n_layers,[[n_neurons_intermediate,n_neurons_output]]]))
    layer_types = list(itertools.chain(*[['fc']*(len(h_sizes)-2), ['fc'], ['fc']]))
    activation = list(itertools.chain(*[[nn.Tanh()]*(len(h_sizes)-1), [nn.Identity()]]))

    return h_sizes, layer_types, activation

########### Import the Neural Network class ###########

DICT_MEAN = {
    "DATA_SPLIT": {
        "shuffle": True,
        "test_size": 0.1,
    },
    "DATA_LOADER": {
        "batch_sizes": [300000,50000,3000,500],
        "X_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "y_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "transform_data": False,
        "shuffle_every_epoch": True
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
        "set_num_threads": 0,
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
        "batch_sizes": [300000,50000,3000,500],
        "X_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "y_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "transform_data": False,
        "shuffle_every_epoch": True
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
        "set_num_threads": 0,
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
        "batch_sizes": [300000,50000,3000,500],
        "X_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "y_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "transform_data": False,
        "shuffle_every_epoch": True
    },
    "NET_DEF": {
        "n_neurons_intermediate": 8,
        "n_layers": 4,
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
        "set_num_threads": 0,
        "patience": 10,
        "factor": 0.5,
        "verbose": True  
    }
}