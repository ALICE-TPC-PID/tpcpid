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
import random
import numpy as np
import json
from copy import deepcopy
import onnxruntime as ort
import torch
import time as timesleep

from sklearn.model_selection import train_test_split

########### Load the configurations.json ###########

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="configuration.json", help="Path to the configuration file")
parser.add_argument("-trm", "--train-mode", default='MEAN', help="Mode in which training is run. Options are: MEAN, SIGMA or FULL")
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
exectime = dt.datetime.now().time()
job_id = os.environ.get('SLURM_JOB_ID', 'local_run')
verbose = (int(os.environ.get("SLURM_PROCID", "0")) == 0)

if verbose:
    LOG.info("SLURM job ID: " + str(job_id))
    LOG.info("Date (dd/mm/yyyy): " + date.strftime('%02d/%02m/%04Y'))
    LOG.info("Time (hh/mm/ss): " + exectime.strftime('%02H:%02M:%02S'))
    LOG.info("Output-folder: " + output_folder)

########### Import the data ###########

LOG.info("Loading data from ROOT file " + data_file)
if data_file.split(".")[-1] == "root":
    cload = load_tree()
    labels, fit_data = cload.load(use_vars=LABELS_X + LABELS_Y, path=data_file, load_latest=True, dtype=np.float32)
elif data_file.split(".")[-1] == "txt":
    labels = np.asarray(LABELS_X + LABELS_Y)
    fit_data = np.loadtxt(data_file, dtype=np.float32, ndmin=2)
else:
    LOG.info("Error: Allowed file type is one of ['ROOT','TXT'].")
    exit()

labels = np.array(labels).astype(str)
fit_data = np.asarray(fit_data, dtype=np.float32)
# Match the configured feature order, rather than ROOT's branch order.
columns = {label: i for i, label in enumerate(labels)}
X = np.ascontiguousarray(fit_data[:, [columns[label] for label in LABELS_X]])
if len(LABELS_Y) != 2:
    raise ValueError('Expected signal and inverse expected dEdx labels')
y = fit_data[:, columns[LABELS_Y[0]]] * fit_data[:, columns[LABELS_Y[1]]]
del fit_data
example_data = torch.from_numpy(X[:1].copy())
# Use one seed for dataset preparation, splitting, initialization and shuffling.
# Legacy per-stage keys remain supported when explicitly configured.
legacy_seed = CONFIG['trainNeuralNetOptions'].get(
    'training_seed', CONFIG['trainNeuralNetOptions'].get('split_seed', 42))
random_seed = int(CONFIG.get('settings', {}).get('random_seed', legacy_seed))
split_seed = training_seed = random_seed
random.seed(training_seed)
np.random.seed(training_seed)
torch.manual_seed(training_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(training_seed)
cpu_threads = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
torch.set_num_threads(cpu_threads)


def predict_batched(network, values, batch_size=65536):
    network.eval()
    output = np.empty(len(values), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            output[start:stop] = network(torch.from_numpy(values[start:stop])).numpy().reshape(-1)
    return output


if args.train_mode == 'MEAN':

    dict_config = deepcopy(nnconfig.DICT_MEAN)
    dict_config["NET_DEF"]["n_neurons_input"] = len(LABELS_X)
    dict_config["NET_TRAINING"]["epochs"] = EPOCHS
    dict_config["NET_TRAINING"]["loss_function"] = weighted_mse_loss

    y = y.reshape(-1, 1)

elif args.train_mode == "SIGMA":

    dict_config = deepcopy(nnconfig.DICT_SIGMA)
    dict_config["NET_DEF"]["n_neurons_input"] = len(LABELS_X)
    dict_config["NET_TRAINING"]["epochs"] = EPOCHS
    dict_config["NET_TRAINING"]["loss_function"] = weighted_mse_loss

    mean_config = deepcopy(nnconfig.DICT_MEAN)
    mean_config["NET_DEF"]["n_neurons_input"] = len(LABELS_X)

    net_mean = NN(nnconfig.model(mean_config))
    net_mean.load_net(
        output_folder + "/networks/network_mean/net_torch_mean.pt",
        map_location="cpu"
    )

    mean = predict_batched(net_mean, X)
    del net_mean

    diff_mean = np.abs(y - mean)
    y = (diff_mean * np.sqrt(np.pi / 2.)).reshape(-1, 1)
    del mean, diff_mean

elif args.train_mode == "FULL":

    dict_config = deepcopy(nnconfig.DICT_FULL)
    dict_config["NET_DEF"]["n_neurons_input"] = len(LABELS_X)
    dict_config["NET_TRAINING"]["epochs"] = EPOCHS
    dict_config["NET_TRAINING"]["loss_function"] = weighted_mse_loss

    mean_config = deepcopy(nnconfig.DICT_MEAN)
    mean_config["NET_DEF"]["n_neurons_input"] = len(LABELS_X)

    sigma_config = deepcopy(nnconfig.DICT_SIGMA)
    sigma_config["NET_DEF"]["n_neurons_input"] = len(LABELS_X)

    net_mean = NN(nnconfig.model(mean_config))
    net_mean.load_net(
        output_folder + "/networks/network_mean/net_torch_mean.pt",
        map_location="cpu"
    )

    net_sigma = NN(nnconfig.model(sigma_config))
    net_sigma.load_net(
        output_folder + "/networks/network_sigma/net_torch_sigma.pt",
        map_location="cpu"
    )

    mean = predict_batched(net_mean, X)
    sigma = predict_batched(net_sigma, X)
    del net_mean, net_sigma

    y = np.column_stack((mean, mean + sigma))
    del mean, sigma

else:
    LOG.info("Unknown args.train_mode! Please select 'MEAN', 'SIGMA' or 'FULL'.")
    exit()

##### Network training #####

NeuralNet = NN(nnconfig.model(dict_config))

### data preparation
split_options = dict(dict_config["DATA_SPLIT"])
if split_options.get('random_state') is None:
    split_options['random_state'] = split_seed
X_train, X_test, y_train, y_test = train_test_split(X, y, **split_options)
del X, y
# Bind before any optional CUDA-resident data preparation.
if torch.cuda.is_available():
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', '0')))
    torch.cuda.set_device(0 if torch.cuda.device_count() == 1 else local_rank)
loader_options = dict(dict_config["DATA_LOADER"])
loader_options.setdefault('seed', random_seed)
data = DataLoading(
    [X_train, y_train],
    [X_test, y_test],
    **loader_options,
    verbose=(int(os.environ.get("SLURM_PROCID", "0")) == 0)
)

del X_train, X_test, y_train, y_test

### evaluate training and validation loss over epochs
NeuralNet.training(data, **dict_config["NET_TRAINING"])

### save the network and the losses
if str(args.train_mode) in ["MEAN", "SIGMA", "FULL"]:
    NeuralNet.eval()
    if save_as_pt == "True":
        NeuralNet.save_net(
            path=output_folder + '/networks/network_' + str(args.train_mode).lower() + '/net_torch_' + str(args.train_mode).lower() + '.pt',
            avoid_q=True
        )
    if save_as_onnx == "True":
        NeuralNet.save_onnx(
            example_data=example_data,
            path=output_folder + '/networks/network_' + str(args.train_mode).lower() + '/net_onnx_' + str(args.train_mode).lower() + '.onnx'
        )
        NeuralNet.check_onnx(
            path=output_folder + '/networks/network_' + str(args.train_mode).lower() + '/net_onnx_' + str(args.train_mode).lower() + '.onnx'
        )
    if save_loss_in_files == "True":
        NeuralNet.save_losses(
            path=[
                output_folder + '/networks/network_' + str(args.train_mode).lower() + '/training_loss_' + str(args.train_mode).lower() + '.txt',
                output_folder + '/networks/network_' + str(args.train_mode).lower() + '/validation_loss_' + str(args.train_mode).lower() + '.txt'
            ]
        )

elif str(args.train_mode) == "ENSEMBLE":
    NeuralNet.eval()
    if save_as_pt == "True":
        NeuralNet.save_net(
            path=output_folder + '/networks/network_' + str(args.train_mode).lower() + '/net_torch_ensemble_' + str(job_id) + '.pt',
            avoid_q=True
        )
    if save_as_onnx == "True":
        NeuralNet.save_onnx(
            example_data=example_data,
            path=output_folder + '/networks/network_' + str(args.train_mode).lower() + '/net_onnx_ensemble_' + str(job_id) + '.onnx'
        )
        NeuralNet.check_onnx(
            path=output_folder + '/networks/network_' + str(args.train_mode).lower() + '/net_onnx_ensemble_' + str(job_id) + '.onnx'
        )
    if save_loss_in_files == "True":
        NeuralNet.save_losses(
            path=[
                output_folder + '/networks/network_' + str(args.train_mode).lower() + '/training_loss_' + str(job_id) + '.txt',
                output_folder + '/networks/network_' + str(args.train_mode).lower() + '/validation_loss_' + str(job_id) + '.txt'
            ]
        )

if verbose:
    LOG.info("Done!")
