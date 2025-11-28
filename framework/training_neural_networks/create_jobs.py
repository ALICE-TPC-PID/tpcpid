"""
File: create_jobs.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
Description: This file creates the jobs directory based on a configuration file (typcially ./config.json)
"""

import os, sys
import json
import glob
import argparse

#################################

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, help="Path to configuration file")
parser.add_argument("-avoid-q", "--avoid-question", type=int, default=0, help="Avoid copying QA scripts")
args = parser.parse_args()

config = args.config
with open(config, 'r') as config_file:
    CONFIG = json.load(config_file)
sys.path.append(os.path.join(CONFIG["output"]["general"]["base_folder"], "framework"))

from base.config_tools import *

### directory settings
toplevel        = CONFIG["output"]["general"]["path"]
output_folder   = CONFIG["output"]["general"]["training"]
data_file       = CONFIG["output"]["shiftNsigma"]["training_data"]

### network settings
execution_mode  = CONFIG["trainNeuralNetOptions"]["execution_mode"]
training_file   = CONFIG["trainNeuralNetOptions"]["training_file"]
qa_file         = CONFIG["trainNeuralNetOptions"]["qa_file"]
num_networks    = CONFIG["trainNeuralNetOptions"]["num_networks"]
enable_qa       = CONFIG["trainNeuralNetOptions"]["enable_qa"]


if os.path.exists(output_folder):
    if args.avoid_question:
        os.system('rm -rf {0}'.format(output_folder))
        os.makedirs(output_folder)
    else:
        response = input("Jobs directory ({}) exists.  Overwrite it? (y/n) ".format(output_folder))
        if response == 'y':
            os.system('rm -rf {0}'.format(output_folder))
            os.makedirs(output_folder)
        else:
            print("Stopping macro!")
            sys.exit(1)

for file in glob.glob(data_file, recursive=True):
    if os.path.isfile(file):
        file_type = file.split(".")[-1]
        os.makedirs(os.path.join(output_folder, 'networks'))
        if ("RUN12" in execution_mode):
            os.makedirs(os.path.join(output_folder, 'networks', 'network_run12'))
        if ("MEAN" in execution_mode) or ("FULL" in execution_mode):
            os.makedirs(os.path.join(output_folder, 'networks', 'network_mean'))
        if ("SIGMA" in execution_mode) or ("FULL" in execution_mode):
            os.makedirs(os.path.join(output_folder, 'networks', 'network_sigma'))
        if "FULL" in execution_mode:
            os.makedirs(os.path.join(output_folder, 'networks', 'network_full'))
        if "ENSEMBLE" in execution_mode:
            for i in range(num_networks):
                os.makedirs(os.path.join(output_folder, 'networks', 'network_' + str(i)))

os.system('cp {0} {1}'.format(CONFIG["trainNeuralNetOptions"]["configuration"], os.path.join(output_folder, 'configurations.py')))
CONFIG["trainNeuralNetOptions"]["configuration"] = os.path.join(output_folder, 'configurations.py')
write_config(CONFIG, path=os.path.join(toplevel, 'configuration.json'))
