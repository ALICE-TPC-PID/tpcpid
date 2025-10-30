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

mask_X = []
mask_y = []
for l in labels:
    mask_X.append(l in configurations.LABELS_X)
    mask_y.append(l in configurations.LABELS_Y)

X = fit_data[:,mask_X]
y = (fit_data[:,mask_y].T[0]*fit_data[:,mask_y].T[1])


def run_network(data, ort_session, hardware=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    return np.array(ort_session.run(None, {'input': (torch.tensor(data).float().to(hardware)).numpy()})[0])

if args.train_mode=='MEAN':

    dict_config = configurations.DICT_MEAN
    y = y.reshape(-1,1)

elif args.train_mode=="SIGMA":

    dict_config = configurations.DICT_SIGMA
    net_mean = torch.load(args.local_training_dir+"/networks/network_mean/net_torch_mean.pt", map_location=torch.device('cpu'))
    mean = net_mean(torch.tensor(X).float()).detach().numpy().flatten()

    diff_mean = np.abs(y-mean)
    y = (diff_mean*np.sqrt(np.pi/2.)).reshape(-1,1)

elif args.train_mode=="FULL":
    
    dict_config = configurations.DICT_FULL

    #ort_sess_mean = ort.InferenceSession(training_dir+"/"+output_folder+"/networks/network_mean/net_onnx_mean.onnx", providers=['MIGraphXExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    #ort_sess_sig_pos = ort.InferenceSession(training_dir+"/"+output_folder+"/networks/network_sigma_pos/net_onnx_sigma_pos.onnx", providers=['MIGraphXExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    #ort_sess_sig_neg = ort.InferenceSession(training_dir+"/"+output_folder+"/networks/network_sigma_neg/net_onnx_sigma_neg.onnx", providers=['MIGraphXExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

    ### Creating mean, sigma_pos and sigma_neg for each datapoint in V0 sample: Populating phase-space
    ### This could be replaced by a regular grid-sampling for better population and faster training...
    #mean = run_network(X, ort_sess_mean).T[0]
    #sig_pos = run_network(X, ort_sess_sig_pos).T[0]
    #sig_neg = run_network(X, ort_sess_sig_neg).T[0]

    # mean = np.loadtxt('/lustre/alice/users/csonnab/TPC/NeuralNetworks/TrainingNetworks/TEMP/mean.txt').astype(float).reshape(-1,1)#[:limit]
    # sig_pos = np.loadtxt('/lustre/alice/users/csonnab/TPC/NeuralNetworks/TrainingNetworks/TEMP/sig_pos.txt').astype(float).reshape(-1,1)#[:limit]
    # sig_neg = np.loadtxt('/lustre/alice/users/csonnab/TPC/NeuralNetworks/TrainingNetworks/TEMP/sig_neg.txt').astype(float).reshape(-1,1)#[:limit]

    net_mean = torch.load(args.local_training_dir+"/networks/network_mean/net_torch_mean.pt", map_location=torch.device('cpu'))
    net_sigma = torch.load(args.local_training_dir+"/networks/network_sigma/net_torch_sigma.pt", map_location=torch.device('cpu'))
    
    mean = torch.flatten(net_mean(torch.tensor(X).float())).detach().numpy()
    sigma = torch.flatten(net_sigma(torch.tensor(X).float())).detach().numpy()
    
    y = np.vstack((mean, mean+sigma)).T

    ### net_mean = torch.load(args.local_training_dir+"/networks/network_mean/net_torch_mean.pt", map_location=torch.device(hardware))
    ### mean_data = torch.tensor(X).float().to(hardware)
    ### y = torch.tensor( np.vstack(( y, np.abs(y - (net_mean(mean_data).float().cpu().detach().numpy()).flatten()*np.sqrt(np.pi/2.)) )).T ).float().to(hardware)

else:

    print("Unknown args.train_mode! Please select 'MEAN', 'SIGMA' or 'FULL'.")
    exit()


##### Network training #####

H_SIZES, LAYER_TYPES, ACTIVATION = configurations.network_def(**dict_config["NET_DEF"])
NeuralNet = NN(General_NN(params = H_SIZES, layer_types = LAYER_TYPES, act_func =ACTIVATION, **dict_config["NET_SETTINGS"]))

### data preparation
X_train, X_test, y_train, y_test = train_test_split(X,y,**dict_config["DATA_SPLIT"])
data = DataLoading([X_train, y_train], [X_test, y_test], **dict_config["DATA_LOADER"])

### evaluate training and validation loss over epochs                   
NeuralNet.training(data, **dict_config["NET_TRAINING"])

### save the network and the losses
if str(args.train_mode) in ["MEAN", "SIGMA", "FULL"]:
    NeuralNet.eval()
    if save_as_pt == "True":
        NeuralNet.save_net(path=args.local_training_dir+'/networks/network_'+str(args.train_mode).lower()+'/net_torch_'+str(args.train_mode).lower()+'.pt',avoid_q=True)
        #NeuralNet.save_jit_script(path=args.local_training_dir+'/networks/network_'+str(args.train_mode).lower()+'/net_'+str(args.train_mode).lower()+'_jit.pt')
    if save_as_onnx == "True":
        NeuralNet.save_onnx(example_data=torch.tensor(np.array([X[0]]),requires_grad=True).float(),
                            path=args.local_training_dir+'/networks/network_'+str(args.train_mode).lower()+'/net_onnx_'+str(args.train_mode).lower()+'.onnx')
        NeuralNet.check_onnx(path=args.local_training_dir+'/networks/network_'+str(args.train_mode).lower()+'/net_onnx_'+str(args.train_mode).lower()+'.onnx')
    if save_loss_in_files == "True":
        NeuralNet.save_losses(path=[args.local_training_dir+'/networks/network_'+str(args.train_mode).lower()+'/training_loss_'+str(args.train_mode).lower()+'.txt',
                                    args.local_training_dir+'/networks/network_'+str(args.train_mode).lower()+'/validation_loss_'+str(args.train_mode).lower()+'.txt'])

elif str(args.train_mode)=="ENSEMBLE":
    NeuralNet.eval()
    if save_as_pt == "True":
        NeuralNet.save_net(path=args.local_training_dir+'/networks/network_'+str(args.train_mode).lower()+'/net_torch_ensemble_'+str(args.job_id)+'.pt',avoid_q=True)
    if save_as_onnx == "True":
        NeuralNet.save_onnx(example_data=torch.tensor(np.array([X[0]]),requires_grad=True).float(),
                            path=args.local_training_dir+'/networks/network_'+str(args.train_mode).lower()+'/net_onnx_ensemble_'+str(args.job_id)+'.onnx',
                            input_names=configurations.LABELS_X, output_names=[str(args.train_mode)])
        NeuralNet.check_onnx(path=args.local_training_dir+'/networks/network_'+str(args.train_mode).lower()+'/net_onnx_ensemble_'+str(args.job_id)+'.onnx')
    if save_loss_in_files == "True":
        NeuralNet.save_losses(path=[args.local_training_dir+'/networks/network_'+str(args.train_mode).lower()+'/training_loss_'+str(args.job_id)+'.txt',
                                    args.local_training_dir+'/networks/network_'+str(args.train_mode).lower()+'/validation_loss_'+str(args.job_id)+'.txt'])


print("\nDone!")