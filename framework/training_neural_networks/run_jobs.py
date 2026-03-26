"""
File: run_jobs.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
Description: This file executes the scripts in the job folder created with the create_jobs.py macro
"""

import os
import sys
import json
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, help="Path to configuration file")
parser.add_argument("-ci", "--ci-run", type=int, default=0, help="Run in CI mode (stream output to terminal)")
args = parser.parse_args()

config = args.config
with open(config, 'r') as config_file:
    CONFIG = json.load(config_file)

sys.path.append(CONFIG['settings']['framework'] + "/framework")
from base import *
LOG = logger(min_severity=CONFIG["process"].get("severity", "DEBUG"), task_name="run_jobs")

### execution settings
output_folder = CONFIG["output"]["general"]["training"]
execution_mode = CONFIG["trainNeuralNetOptions"]["execution_mode"]
base_folder = CONFIG["settings"]["framework"]
scheduler = determine_scheduler(scheduler=CONFIG["trainNeuralNetOptions"].get("scheduler", None), verbose=False)
CONFIG["trainNeuralNetOptions"]["scheduler"] = scheduler
write_config(CONFIG, args.config)


def parse_first_level(directory):
    dirs = []
    for elem in os.listdir(directory):
        full_path = os.path.join(directory, elem)
        if os.path.isdir(full_path) and elem != "__pycache__":
            dirs.append(full_path)
    return dirs


def run_command(cmd, name):
    LOG.info(f"Running {name}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)

    if proc.returncode != 0:
        LOG.error(f"{name} failed with return code {proc.returncode}")
        sys.exit(proc.returncode)

    LOG.info(f"{name} finished successfully")


data_dirs = parse_first_level(output_folder)

training_dir = os.path.join(base_folder, "framework", "training_neural_networks")
shell_script_creation = os.path.join(training_dir, "shell_script_creation.py")
run_job_single_sigma = os.path.join(training_dir, "run_job_single_sigma.py")

training_script = os.path.join(training_dir, CONFIG["trainNeuralNetOptions"]["training_file"])

qa_script = os.path.join(training_dir, CONFIG["trainNeuralNetOptions"]["qa_file"])

run_command(
    [
        sys.executable,
        shell_script_creation,
        "--config", args.config,
        "--job-script", training_script,
    ],
    "shell_script_creation (training)"
)

run_command(
    [
        sys.executable,
        shell_script_creation,
        "--config", args.config,
        "--job-script", qa_script,
        "--training-mode", "QA",
    ],
    "shell_script_creation (QA)"
)

for tr_dir in data_dirs:
    run_command(
        [
            sys.executable,
            run_job_single_sigma,
            "--config", args.config,
            "--ci-run", str(args.ci_run),
        ],
        f"run_job_single_sigma for {tr_dir}"
    )