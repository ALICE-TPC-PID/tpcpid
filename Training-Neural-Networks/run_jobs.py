"""
File: run_jobs.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
Description: This file executes the scripts in the job folder created with the create_jobs.py macro
"""

import json
import os
import subprocess
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils.config_tools import (
    add_name_and_path,
    read_config,
    write_config,
)

verbose = False
CONFIG = read_config()

### execution settings
output_folder   = CONFIG["output"]["general"]["training"]
scheduler = CONFIG["trainNeuralNetOptions"]["scheduler"]

### network settings
execution_mode  = CONFIG["trainNeuralNetOptions"]["execution_mode"]

def determine_scheduler():

    def test_env(cmd):
        return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell=True)==0
        
    if not scheduler:
        avail_schedulers = []
        slurm_env = test_env("squeue -u {}".format(os.environ.get("USER")))
        condor_env = test_env("condor_q")
        if slurm_env:
            avail_schedulers.append("slurm")
        if condor_env:
            avail_schedulers.append("htcondor")
        if verbose > 0:
            print("The following schedulers are available: ", avail_schedulers)
            print(avail_schedulers[0], "is picked for submission\n")
        return avail_schedulers[0]
    else:
        if verbose > 0:
            print(scheduler, "is picked for submission\n")
        return scheduler


def parse_first_level(dir):
    dirs = list()
    for elem in os.listdir(dir):
        if os.path.isdir(dir+"/"+elem) and elem!="__pycache__":
            dirs.append(dir+"/"+elem)
    return dirs

data_dirs = parse_first_level(output_folder)
scheduler = determine_scheduler()

os.system("python3 {0}/shell_script_creation.py --job-script {1} --scheduler {2} ".format(output_folder, output_folder+"/train.py", scheduler))
os.system("python3 {0}/shell_script_creation.py --job-script {1} --scheduler {2}  --training-mode QA".format(output_folder, output_folder+"/training_qa.py", scheduler))
for tr_dir in data_dirs:
    os.system("python3 run_job_single_sigma.py --current-dir {0} --scheduler {1} ".format(tr_dir, scheduler))