"""
File: train_single_sigma.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
"""

import sys
import os
import argparse
import datetime
import numpy as np
import json
import onnxruntime as ort
import torch
import glob

from sklearn.model_selection import train_test_split

########### Load the configurations from config.json ###########

### Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-trm", "--train-mode", default='MEAN', help="Mode in which training is run. Options are: MEAN, SIGMA or FULL") # choices=['MEAN', 'SIGMA', 'FULL', 'ENSEMBLE'] fails (?)
parser.add_argument("-jid", "--job-id", default=-1, help="ID of the submitted slurm job")
parser.add_argument("-locdir", "--local-training-dir", default=".", help="Local directory for training of the neural network")
args = parser.parse_args()

### External json settings
configs_file = open("config.json", "r")
CONF = json.load(configs_file)

### directory settings
training_dir        = CONF["directories"]["training_dir"]
output_folder       = CONF["directories"]["output_folder"]
data_file           = CONF["directories"]["data_file"]

### network settings
train_mode          = CONF["network"]["execution_mode"]
num_networks        = CONF["network"]["num_networks"]
save_as_pt          = CONF["network"]["save_as_pt"]
save_as_onnx        = CONF["network"]["save_as_onnx"]
save_loss_in_files  = CONF["network"]["save_loss_in_files"]

configs_file.close()

training_file = glob.glob(args.local_training_dir+"/*.*")[0]

########### Print the date, time and location for identification ###########

date = datetime.datetime.now().date()
time = datetime.datetime.now().time()
print("Info:\n")
print("SLURM job ID:", args.job_id)
print("Date (dd/mm/yyyy):",date.strftime('%02d/%02m/%04Y'))
print("Time (hh/mm/ss):", time.strftime('%02H:%02M:%02S'))
print("Output-folder:", training_dir+"/"+output_folder+"/"+args.local_training_dir)

hardware = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########### Import the Neural Network class ###########

NN_dir = training_dir + "/../Neural-Network-Class/NeuralNetworkClasses"
sys.path.append(NN_dir)

from extract_from_root import load_tree
from dataset_loading import DataLoading
from NN_class import General_NN, NN

import configurations

########### Import the data ###########

if training_file.split(".")[-1] == "root":
    cload = load_tree()
    labels, fit_data = cload.load(use_vars=configurations.LABELS_X+configurations.LABELS_Y, path=training_file)
elif training_file.split(".")[-1] == "txt":
    labels, fit_data = np.loadtxt(training_file, dtype='S')
else:
    print("Error: Allowed file type is one of ['ROOT','TXT'].")

labels = np.array(labels).astype(str)
fit_data = np.array(fit_data).astype(float)

reorder_index = []
for lab in [*configurations.LABELS_X,*configurations.LABELS_Y]:
    reorder_index.append(np.where(labels==lab)[0][0])
reorder_index = np.array(reorder_index)
fit_data = fit_data[:,reorder_index]
labels = labels[reorder_index]

mask_X = []
mask_y_mean, mask_y_sigma = [], []
for l in labels:
    mask_X.append(l in configurations.LABELS_X)
    mask_y_mean.append(l in configurations.LABELS_Y[0])
    mask_y_sigma.append(l in configurations.LABELS_Y[1])

X = fit_data[:,mask_X]
y_mean = fit_data[:,mask_y_mean]
y_sigma = fit_data[:,mask_y_sigma] + fit_data[:,mask_y_mean]

dict_config = configurations.DICT_FULL
y = np.hstack((y_mean, y_sigma))


##### Network training #####

H_SIZES, LAYER_TYPES, ACTIVATION = configurations.network_def(**dict_config["NET_DEF"])
NeuralNet = NN(General_NN(params = H_SIZES, layer_types = LAYER_TYPES, act_func =ACTIVATION, **dict_config["NET_SETTINGS"]))

### data preparation
X_train, X_test, y_train, y_test = train_test_split(X,y,**dict_config["DATA_SPLIT"])
data = DataLoading([X_train, y_train], [X_test, y_test], **dict_config["DATA_LOADER"])

### evaluate training and validation loss over epochs                   
NeuralNet.training(data, **dict_config["NET_TRAINING"])

### save the network and the losses
NeuralNet.eval()
if save_as_pt == "True":
    NeuralNet.save_net(path=args.local_training_dir+'/networks/network_'+str(train_mode).lower()+'/net_torch_'+str(train_mode).lower()+'.pt',avoid_q=True)
    #NeuralNet.save_jit_script(path=training_dir+"/"+output_folder+"/"+tr_dir+'/networks/network_'+str(train_mode).lower()+'/net_'+str(train_mode).lower()+'_jit.pt')
if save_as_onnx == "True":
    NeuralNet.save_onnx(example_data=torch.tensor(np.array([X[0]]),requires_grad=True).float(),
                        path=args.local_training_dir+'/networks/network_'+str(train_mode).lower()+'/net_onnx_'+str(train_mode).lower()+'.onnx')
    NeuralNet.check_onnx(path=args.local_training_dir+'/networks/network_'+str(train_mode).lower()+'/net_onnx_'+str(train_mode).lower()+'.onnx')
if save_loss_in_files == "True":
    NeuralNet.save_losses(path=[args.local_training_dir+'/networks/network_'+str(train_mode).lower()+'/training_loss_'+str(train_mode).lower()+'.txt',
                                args.local_training_dir+'/networks/network_'+str(train_mode).lower()+'/validation_loss_'+str(train_mode).lower()+'.txt'])