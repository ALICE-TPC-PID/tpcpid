"""
File: run_jobs.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
Description: This file executes the scripts in the job folder created with the create_jobs.py macro
"""

import os, sys, json
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, help="Path to configuration file")
args = parser.parse_args()

config = args.config
with open(config, 'r') as config_file:
    CONFIG = json.load(config_file)
sys.path.append(CONFIG['settings']['framework'] + "/framework")
from base import *

### execution settings
output_folder                                   = CONFIG["output"]["general"]["training"]
execution_mode                                  = CONFIG["trainNeuralNetOptions"]["execution_mode"]
base_folder                                     = CONFIG['settings']['framework']
scheduler                                       = determine_scheduler(verbose=False)
CONFIG["trainNeuralNetOptions"]["scheduler"]    = scheduler
write_config(CONFIG, args.config)

def parse_first_level(dir):
    dirs = list()
    for elem in os.listdir(dir):
        if os.path.isdir(dir+"/"+elem) and elem!="__pycache__":
            dirs.append(dir+"/"+elem)
    return dirs

data_dirs = parse_first_level(output_folder)

os.system("python3 {0}/shell_script_creation.py --config {1} --job-script {2} ".format(base_folder+"/framework/training_neural_networks", args.config, base_folder+"/framework/training_neural_networks/"+CONFIG["trainNeuralNetOptions"]["training_file"]))
os.system("python3 {0}/shell_script_creation.py --config {1} --job-script {2} --training-mode QA".format(base_folder+"/framework/training_neural_networks", args.config, base_folder+"/framework/training_neural_networks/"+CONFIG["trainNeuralNetOptions"]["qa_file"]))
for tr_dir in data_dirs:
    os.system("python3 {0}/run_job_single_sigma.py --config {1}".format(base_folder+"/framework/training_neural_networks", args.config))