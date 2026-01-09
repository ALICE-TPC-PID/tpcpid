"""
File: train_single_sigma.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
"""

import sys
import os
import argparse
import datetime as dt
import numpy as np
import json
import onnxruntime as ort
import torch

from sklearn.model_selection import train_test_split

########### Load the configurations.json ###########

### Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="configuration.json", help="Path to the configuration file")
parser.add_argument("-trm", "--train-mode", default='MEAN', help="Mode in which training is run. Options are: MEAN, SIGMA or FULL") # choices=['MEAN', 'SIGMA', 'FULL', 'ENSEMBLE'] fails (?)
args = parser.parse_args()

with open(args.config, 'r') as config_file:
    CONFIG = json.load(config_file)
sys.path.append(CONFIG['settings']['framework'] + "/framework")
from base import *
from neural_network_class.NeuralNetworkClasses.extract_from_root import *
from neural_network_class.NeuralNetworkClasses.dataset_loading import *
from neural_network_class.NeuralNetworkClasses.NN_class import *

LOG = logger(min_severity=CONFIG["process"].get("severity", "DEBUG"), task_name="train_single_sigma")

nnconfig = import_from_path(CONFIG["trainNeuralNetOptions"]["configuration"])

### directory settings
output_folder       = CONFIG["output"]["general"]["training"]
data_file           = CONFIG["output"]["createTrainingDataset"]["training_data"]
train_mode          = CONFIG["trainNeuralNetOptions"]["execution_mode"]
num_networks        = CONFIG["trainNeuralNetOptions"]["num_networks"]
training_file       = CONFIG["trainNeuralNetOptions"]["training_file"]
save_as_pt          = CONFIG["trainNeuralNetOptions"]["save_as_pt"]
save_as_onnx        = CONFIG["trainNeuralNetOptions"]["save_as_onnx"]
save_loss_in_files  = CONFIG["trainNeuralNetOptions"]["save_loss_in_files"]

LABELS_X        = CONFIG['createTrainingDatasetOptions']['labels_x']
LABELS_Y        = CONFIG['createTrainingDatasetOptions']['labels_y']
BB_PARAMS       = CONFIG['output']['fitBBGraph']['BBparameters']
EPOCHS          = CONFIG['trainNeuralNetOptions']['numberOfEpochs']

########### Print the date, time and location for identification ###########

date = dt.datetime.now().date()
time = dt.datetime.now().time()
job_id = os.environ.get('SLURM_JOB_ID', 'local_run')
verbose = (int(os.environ.get("SLURM_PROCID", "0"))==0)

if verbose:
    LOG.info("SLURM job ID: " + str(job_id))
    LOG.info("Date (dd/mm/yyyy): " + date.strftime('%02d/%02m/%04Y'))
    LOG.info("Time (hh/mm/ss): " + time.strftime('%02H:%02M:%02S'))
    LOG.info("Output-folder: " + output_folder)

########### Import the data ###########

if data_file.split(".")[-1] == "root":
    cload = load_tree()
    labels, fit_data = cload.load(use_vars=LABELS_X+LABELS_Y, path=data_file, load_latest=False)
elif data_file.split(".")[-1] == "txt":
    labels, fit_data = np.loadtxt(data_file, dtype='S')
else:
    LOG.info("Error: Allowed file type is one of ['ROOT','TXT'].")


labels = np.array(labels).astype(str)
fit_data = np.array(fit_data).astype(float)

mask_X = []
mask_y = []
for l in labels:
    mask_X.append(l in LABELS_X)
    mask_y.append(l in LABELS_Y)

X = fit_data[:,mask_X]
y = (fit_data[:,mask_y].T[0]*fit_data[:,mask_y].T[1])


def run_network(data, ort_session, hardware=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    return np.array(ort_session.run(None, {'input': (torch.tensor(data).float().to(hardware)).numpy()})[0])

if args.train_mode=='MEAN':

    dict_config = nnconfig.DICT_MEAN
    dict_config["NET_DEF"]["n_neurons_input"] = len(LABELS_X)
    dict_config["NET_TRAINING"]["epochs"] = EPOCHS
    dict_config["NET_TRAINING"]["loss_function"] = weighted_mse_loss

    y = y.reshape(-1,1)

elif args.train_mode=="SIGMA":

    dict_config = nnconfig.DICT_SIGMA
    dict_config["NET_DEF"]["n_neurons_input"] = len(LABELS_X)
    dict_config["NET_TRAINING"]["epochs"] = EPOCHS
    dict_config["NET_TRAINING"]["loss_function"] = weighted_mse_loss

    net_mean = torch.load(output_folder+"/networks/network_mean/net_torch_mean.pt", map_location=torch.device('cpu'))
    mean = net_mean(torch.tensor(X).float()).detach().numpy().flatten()

    diff_mean = np.abs(y-mean)
    y = (diff_mean*np.sqrt(np.pi/2.)).reshape(-1,1)

elif args.train_mode=="FULL":

    dict_config = nnconfig.DICT_FULL
    dict_config["NET_DEF"]["n_neurons_input"] = len(LABELS_X)
    dict_config["NET_TRAINING"]["epochs"] = EPOCHS
    dict_config["NET_TRAINING"]["loss_function"] = weighted_mse_loss

    net_mean = torch.load(output_folder+"/networks/network_mean/net_torch_mean.pt", map_location=torch.device('cpu'))
    net_sigma = torch.load(output_folder+"/networks/network_sigma/net_torch_sigma.pt", map_location=torch.device('cpu'))

    mean = torch.flatten(net_mean(torch.tensor(X).float())).detach().numpy()
    sigma = torch.flatten(net_sigma(torch.tensor(X).float())).detach().numpy()

    y = np.vstack((mean, mean+sigma)).T

else:

    LOG.info("Unknown args.train_mode! Please select 'MEAN', 'SIGMA' or 'FULL'.")
    exit()


##### Network training #####

H_SIZES, LAYER_TYPES, ACTIVATION = nnconfig.network_def(**dict_config["NET_DEF"])
NeuralNet = NN(General_NN(params = H_SIZES, layer_types = LAYER_TYPES, act_func =ACTIVATION, **dict_config["NET_SETTINGS"]))

### data preparation
X_train, X_test, y_train, y_test = train_test_split(X,y,**dict_config["DATA_SPLIT"])
data = DataLoading([X_train, y_train], [X_test, y_test], **dict_config["DATA_LOADER"], verbose=(int(os.environ.get("SLURM_PROCID", "0"))==0))

### evaluate training and validation loss over epochs
NeuralNet.training(data, **dict_config["NET_TRAINING"])

### save the network and the losses
if str(args.train_mode) in ["MEAN", "SIGMA", "FULL"]:
    NeuralNet.eval()
    if save_as_pt == "True":
        NeuralNet.save_net(path=output_folder+'/networks/network_'+str(args.train_mode).lower()+'/net_torch_'+str(args.train_mode).lower()+'.pt',avoid_q=True)
        #NeuralNet.save_jit_script(path=output_folder+'/networks/network_'+str(args.train_mode).lower()+'/net_'+str(args.train_mode).lower()+'_jit.pt')
    if save_as_onnx == "True":
        NeuralNet.save_onnx(example_data=torch.tensor(np.array([X[0]]),requires_grad=True).float(),
                            path=output_folder+'/networks/network_'+str(args.train_mode).lower()+'/net_onnx_'+str(args.train_mode).lower()+'.onnx')
        NeuralNet.check_onnx(path=output_folder+'/networks/network_'+str(args.train_mode).lower()+'/net_onnx_'+str(args.train_mode).lower()+'.onnx')
    if save_loss_in_files == "True":
        NeuralNet.save_losses(path=[output_folder+'/networks/network_'+str(args.train_mode).lower()+'/training_loss_'+str(args.train_mode).lower()+'.txt',
                                    output_folder+'/networks/network_'+str(args.train_mode).lower()+'/validation_loss_'+str(args.train_mode).lower()+'.txt'])

elif str(args.train_mode)=="ENSEMBLE":
    NeuralNet.eval()
    if save_as_pt == "True":
        NeuralNet.save_net(path=output_folder+'/networks/network_'+str(args.train_mode).lower()+'/net_torch_ensemble_'+str(job_id)+'.pt',avoid_q=True)
    if save_as_onnx == "True":
        NeuralNet.save_onnx(example_data=torch.tensor(np.array([X[0]]),requires_grad=True).float(),
                            path=output_folder+'/networks/network_'+str(args.train_mode).lower()+'/net_onnx_ensemble_'+str(job_id)+'.onnx',
                            input_names=LABELS_X, output_names=[str(args.train_mode)])
        NeuralNet.check_onnx(path=output_folder+'/networks/network_'+str(args.train_mode).lower()+'/net_onnx_ensemble_'+str(job_id)+'.onnx')
    if save_loss_in_files == "True":
        NeuralNet.save_losses(path=[output_folder+'/networks/network_'+str(args.train_mode).lower()+'/training_loss_'+str(job_id)+'.txt',
                                    output_folder+'/networks/network_'+str(args.train_mode).lower()+'/validation_loss_'+str(job_id)+'.txt'])


if verbose:
    LOG.info("Done!")