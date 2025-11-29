import sys, os, json, subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="configuration.json", help="Path to the configuration file")
args = parser.parse_args()

if not os.path.exists(args.config):
    print(f"Configuration file {args.config} not found. Aborting")
    sys.exit(1)
    
with open(args.config, 'r') as config_file:
    CONFIG = json.load(config_file)
sys.path.append(CONFIG['paths']['framework'] + "/framework")
from base import *

LOG = logger.logger("Framework")
LOG.welcome_message()

CONFIG = add_name_and_path(CONFIG)
can_we_continue() # Ask user if they want to continue with the given configuration, recreates the output folders if necessary
create_folders(CONFIG)
config_path = copy_config(CONFIG)

framework_path = CONFIG['paths']['framework']
output_path = CONFIG["output"]["general"]["path"]

exec_script = """#!/bin/bash
#SBATCH --job-name=TPCPID_MASTERJOB                                         # Task name
#SBATCH --chdir={p}                                                         # Working directory on shared storage
#SBATCH --time=10                                                           # Run time limit
#SBATCH --mem=30G                                                           # job memory
#SBATCH --partition=debug                                                   # job partition (debug, main)
#SBATCH --output={l}/run_%j.out                                             # Standard output and error log
#SBATCH --error={l}/run_%j.err                                              # Standard error log

time python3 {p}/run/src/run_framework.py --config $1
""".format(p=framework_path, l=output_path)

script_path = os.path.join(framework_path, 'run/RUN_SLURM.sh')
with open(script_path, 'w') as script_file:
    script_file.write(exec_script)

slurm_out = subprocess.check_output(f"sbatch {script_path} {config_path}", shell=True).decode().strip('\n')
LOG.framework(f"TPCPID_MASTERJOB job submitted successfully. Job ID: {slurm_out.split()[-1]}")