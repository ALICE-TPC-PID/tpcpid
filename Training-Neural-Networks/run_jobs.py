"""
File: run_jobs.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
Description: This file executes the scripts in the job folder created with the create_jobs.py macro
"""

import json
import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config", default="config.json", help="Path to the configuration file, default ./config.json")
parser.add_argument("-sched", "--scheduler", default="", help="System for job-scheduling. Options are [slurm, htcondor]")
parser.add_argument("-BBpath", "--BBparameterPath", default="", help="Path to BB parameters from fitting for specific dataset")
parser.add_argument('-v', '--verbose', action='count', default=0)
args = parser.parse_args()

configs_file = open(args.config, "r")
CONF = json.load(configs_file)

### execution settings
training_dir    = CONF["directories"]["training_dir"]
output_folder   = CONF["directories"]["output_folder"]

### network settings
execution_mode  = CONF["network"]["execution_mode"]

configs_file.close()

def determine_scheduler():

    def test_env(cmd):
        return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell=True)==0
        
    if not args.scheduler:
        avail_schedulers = []
        slurm_env = test_env("squeue -u {}".format(os.environ.get("USER")))
        condor_env = test_env("condor_q")
        if slurm_env:
            avail_schedulers.append("slurm")
        if condor_env:
            avail_schedulers.append("htcondor")
        if args.verbose > 0:
            print("The following schedulers are available: ", avail_schedulers)
            print(avail_schedulers[0], "is picked for submission\n")
        return avail_schedulers[0]
    else:
        if args.verbose > 0:
            print(args.scheduler, "is picked for submission\n")
        return args.scheduler


def parse_first_level(dir):
    dirs = list()
    for elem in os.listdir(dir):
        if os.path.isdir(dir+"/"+elem) and elem!="__pycache__":
            dirs.append(dir+"/"+elem)
    return dirs

data_dirs = parse_first_level(training_dir+"/"+output_folder)
scheduler = determine_scheduler()
BBparameterPath = str(args.BBparameterPath)


os.system("python3 {0}/shell_script_creation.py --job-script {1} --scheduler {2} --config {3}".format(training_dir+"/configurations", training_dir+"/"+output_folder+"/train.py", scheduler, args.config))
os.system("python3 {0}/shell_script_creation.py --job-script {1} --scheduler {2} --config {3} -BB {4} --training-mode QA".format(training_dir+"/configurations", training_dir+"/"+output_folder+"/training_qa.py", scheduler, args.config, BBparameterPath))
for tr_dir in data_dirs:
    os.system("python3 configurations/run_job_single_sigma.py --current-dir {0} --scheduler {1} --config {2}".format(tr_dir, scheduler, args.config))