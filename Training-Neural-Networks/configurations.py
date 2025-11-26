from sklearn import preprocessing
import itertools
import torch.nn as nn
import torch.optim as optim
import sys
import json
from custom_loss_functions import *
from config_tools import (
    add_name_and_path,
    read_config,
    write_config,
)


##### Setting the parameters for the training #####

def network_def(n_neurons_input, n_neurons_intermediate, n_neurons_output, n_layers):

    h_sizes = list(itertools.chain(*[[[n_neurons_input,n_neurons_intermediate]],[[n_neurons_intermediate,n_neurons_intermediate]]*n_layers,[[n_neurons_intermediate,n_neurons_output]]]))
    layer_types = list(itertools.chain(*[['fc']*(len(h_sizes)-2), ['fc'], ['fc']]))
    activation = list(itertools.chain(*[[nn.Tanh()]*(len(h_sizes)-1), [nn.Identity()]]))

    return h_sizes, layer_types, activation

CONFIG = read_config()
LABELS_X = CONFIG['createTrainingDatasetOptions']['labels_x']
LABELS_Y = CONFIG['createTrainingDatasetOptions']['labels_y']
BB_PARAMS = CONFIG['output']['fitBBGraph']['BBparameters']
EPOCHS = CONFIG['trainNeuralNetOptions']['numberOfEpochs']
# LABELS_X = ['fTPCInnerParam', 'fTgl', 'fSigned1Pt', 'fMass', 'fNormMultTPC', 'fNormNClustersTPC','fFt0Occ']
# #LABELS_X: ['fTPCInnerParam', 'fTgl', 'fSigned1Pt', 'fMass', 'fNormNClustersTPC']
# LABELS_Y = ['fTPCSignal', 'fInvDeDxExpTPC']

# BB_PARAMS = [0.228007, 3.93226, 0.0122857, 2.26946, 0.861199, 50, 2.3]


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
        "n_neurons_input": len(LABELS_X),
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
        "epochs": EPOCHS,
        "epochs_ls": [0,30,50,80],
        "weights": False,
        "optimizer": optim.Adam,
        "weight_decay": 0,
        "scheduler": optim.lr_scheduler.ReduceLROnPlateau,
        "loss_function": weighted_mse_loss,
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
        "n_neurons_input": len(LABELS_X),
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
        "epochs": EPOCHS,
        "epochs_ls": [0,30,50,80],
        "weights": False,
        "optimizer": optim.Adam,
        "weight_decay": 0,
        "scheduler": optim.lr_scheduler.ReduceLROnPlateau,
        "loss_function": weighted_mse_loss,
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
        "n_neurons_input": len(LABELS_X),
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
        "epochs": EPOCHS,
        "epochs_ls": [0,30,50,80],
        "weights": False,
        "optimizer": optim.Adam,
        "weight_decay": 0,
        "scheduler": optim.lr_scheduler.ReduceLROnPlateau,
        "loss_function": weighted_mse_loss,
        "learning_rate": 0.001,
        "set_num_threads": 0,
        "patience": 10,
        "factor": 0.5,
        "verbose": True  
    }
}