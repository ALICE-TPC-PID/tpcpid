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

parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config", default="config.json", help="Path to the configuration file, default ./config.json")
args = parser.parse_args()

#################################

configs_file = open(args.config, "r")
CONF = json.load(configs_file)

### directory settings
training_dir    = CONF["directories"]["training_dir"]
output_folder   = CONF["directories"]["output_folder"]
data_file       = CONF["directories"]["data_file"]

### network settings
configurations  = CONF["network"]["configurations_file"]
execution_mode  = CONF["network"]["execution_mode"]
training_file   = CONF["network"]["training_file"]
qa_file         = CONF["network"]["qa_file"]
num_networks    = CONF["network"]["num_networks"]
enable_qa       = CONF["network"]["enable_qa"]

configs_file.close()

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
        exec_dir = os.path.basename(os.path.dirname(file))
        file_type = file.split(".")[-1]
        loc_tr_dir = os.path.join(output_folder, exec_dir)
        # os.makedirs(loc_tr_dir)
        os.makedirs(os.path.join(loc_tr_dir, 'networks'))
        if enable_qa:
            os.makedirs(os.path.join(loc_tr_dir, 'QA'))
        os.system('cp {0} {1}'.format(file, os.path.join(output_folder, exec_dir, 'training_data.' + file_type.lower())))
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

os.system('cp {0} {1}'.format(args.config, os.path.join(output_folder, 'config.json')))
os.system('cp {0} {1}'.format(training_file, os.path.join(output_folder, 'train.py')))
os.system('cp {0} {1}'.format(qa_file, os.path.join(output_folder, 'training_qa.py')))
os.system('cp {0} {1}'.format(configurations, os.path.join(output_folder, 'configurations.py')))
