"""
File: create_jobs.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
Description: This file creates the jobs directory based on a configuration file (typcially ./config.json)
"""

from sys import exit
import os
import json
import glob
import argparse
import pathlib
from config_tools import (
    add_name_and_path,
    read_config,
    write_config,
)


#################################

CONFIG = read_config()

### directory settings
output_folder   = CONFIG["output"]["general"]["training"]
data_file       = CONFIG["output"]["createTrainingDataset"]["training_data"]

### network settings
execution_mode  = CONFIG["trainNeuralNetOptions"]["execution_mode"]
training_file   = CONFIG["trainNeuralNetOptions"]["training_file"]
qa_file         = CONFIG["trainNeuralNetOptions"]["qa_file"]
num_networks    = CONFIG["trainNeuralNetOptions"]["num_networks"]
enable_qa       = CONFIG["trainNeuralNetOptions"]["enable_qa"]


if os.path.exists(output_folder):
    response = input("Jobs directory ({}) exists.  Overwrite it? (y/n) ".format(output_folder))
    if response == 'y':
        os.system('rm -rf {0}'.format(output_folder))
        os.makedirs(output_folder)
    else:
        print("Stopping macro!")
        exit()

for file in glob.glob(data_file, recursive=True):
    if os.path.isfile(file):
        file_type = file.split(".")[-1]
        loc_tr_dir = output_folder
        print(f"[DEBUG]: loc_tr_dir = {loc_tr_dir}")
        # os.makedirs(loc_tr_dir)
        os.makedirs(os.path.join(loc_tr_dir, 'networks'))
        os.system('cp {0} {1}'.format(file, os.path.join(output_folder, 'training_data.' + file_type.lower())))
        if ("RUN12" in execution_mode):
            os.makedirs(os.path.join(loc_tr_dir, 'networks', 'network_run12'))
        if ("MEAN" in execution_mode) or ("FULL" in execution_mode):
            os.makedirs(os.path.join(loc_tr_dir, 'networks', 'network_mean'))
        if ("SIGMA" in execution_mode) or ("FULL" in execution_mode):
            os.makedirs(os.path.join(loc_tr_dir, 'networks', 'network_sigma'))
        if "FULL" in execution_mode:
            os.makedirs(os.path.join(loc_tr_dir, 'networks', 'network_full'))
        if "ENSEMBLE" in execution_mode:
            for i in range(num_networks):
                os.makedirs(os.path.join(loc_tr_dir, 'networks', 'network_' + str(i)))

# os.system('cp {0} {1}'.format(args.config, os.path.join(output_folder, 'config.json')))
os.system('cp {0} {1}'.format(training_file, os.path.join(output_folder, 'train.py')))
os.system('cp {0} {1}'.format(qa_file, os.path.join(output_folder, 'training_qa.py')))
os.system('cp {0} {1}'.format("configurations.py", os.path.join(output_folder, 'configurations.py')))
os.system('cp {0} {1}'.format("run_job_single_sigma.py", os.path.join(output_folder, 'run_job_single_sigma.py')))
os.system('cp {0} {1}'.format("shell_script_creation.py", os.path.join(output_folder, 'shell_script_creation.py')))
os.system('cp {0} {1}'.format("../utils/config_tools.py", os.path.join(output_folder, 'config_tools.py')))
