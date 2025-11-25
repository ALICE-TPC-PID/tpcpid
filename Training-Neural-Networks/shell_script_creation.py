"""
File: shell_script_creation.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
"""

import sys
import os
import json
import argparse
from config_tools import (
    add_name_and_path,
    read_config,
    write_config,
)

from os import path

parser = argparse.ArgumentParser()
parser.add_argument("-jbscript", "--job-script", default=".", help="Path to job script")
parser.add_argument("-trm", "--training-mode", default='MEAN', help="Training mode") #  choices=['MEAN', 'SIGMA', 'FULL', 'ENSEMBLE'] fails (?)
args = parser.parse_args()

job_script      = str(args.job_script)
train_mode      = str(args.training_mode)


CONFIG = read_config()

### directory settings
output_folder   = CONFIG["output"]["general"]["training"]

### scheduler settings
scheduler       = CONFIG["trainNeuralNetOptions"]["scheduler"]
job_dict        = CONFIG["trainNeuralNetOptions"][scheduler]



full_path_out = output_folder
job_dict["chdir"] = full_path_out
job_dict["job_script"] = job_script


if scheduler.lower() == "slurm":

    if train_mode != "QA":

        if job_dict["device"] == "EPN": ### Setup to submit to EPN nodes

            bash_file = open(path.join(full_path_out, "TRAIN.sh".format(train_mode)), "w")
            bash_file.write(
"""#!/bin/bash
#SBATCH --job-name=%(job-name)s                                             # Task name
#SBATCH --chdir=%(chdir)s                                                   # Working directory on shared storage
#SBATCH --time=%(time)s                                                     # Run time limit 
#SBATCH --mem=%(mem)s                                                       # job memory
#SBATCH --partition=prod                                                    # job partition (debug, main)
#SBATCH --mail-type=%(mail-type)s                                           # notify via email
#SBATCH --mail-user=%(mail-user)s                                           # recipient
#SBATCH --gres=gpu:1   		                                                # reservation for GPU

time python3.9 %(job_script)s --train-mode $1 --job-id ${SLURM_JOBID} --local-training-dir $2

""" % job_dict)
            bash_file.close()

        else: ### Setup for GSI batch farm (default)

            if job_dict["device"] == "V100_GPU":
            
                bash_file = open(path.join(full_path_out, "TRAIN.sh".format(train_mode)), "w")
                bash_file.write(
"""#!/bin/bash
#SBATCH --job-name=%(job-name)s                                             # Task name
#SBATCH --chdir=%(chdir)s                                                   # Working directory on shared storage
#SBATCH --time=%(time)s                                                     # Run time limit 
#SBATCH --mem=%(mem)s                                                       # job memory
#SBATCH --partition=long                                                    # job partition (debug, main)
#SBATCH --mail-type=%(mail-type)s                                           # notify via email
#SBATCH --mail-user=%(mail-user)s                                           # recipient
#SBATCH --reservation=nvidia_gpu                                            # reservation for GPU
#SBATCH --constraint=v100    		                                        
#SBATCH --gres=gpu:1                                                        # submit to specific GPU

time singularity exec --nv %(cuda_container)s python3 %(job_script)s --train-mode $1 --job-id ${SLURM_JOBID} --local-training-dir $2

""" % job_dict)
                bash_file.close()
        
            elif job_dict["device"] == "MI100_GPU":
                
                bash_file = open(path.join(full_path_out, "TRAIN.sh".format(train_mode)), "w")
                bash_file.write(
"""#!/bin/bash
#SBATCH --job-name=%(job-name)s                                             # Task name
#SBATCH --chdir=%(chdir)s                                                   # Working directory on shared storage
#SBATCH --time=%(time)s                                                     # Run time limit 
#SBATCH --mem=%(mem)s                                                       # job memory
#SBATCH --partition=gpu                                                     # job partition (debug, main)
#SBATCH --mail-type=%(mail-type)s                                           # notify via email
#SBATCH --mail-user=%(mail-user)s                                           # recipient
#SBATCH --constraint=mi100   		                                        # reservation for GPU
#SBATCH --exclude=lxbk1099

time singularity exec %(rocm_container)s python3 %(job_script)s --train-mode $1 --job-id ${SLURM_JOBID} --local-training-dir $2

""" % job_dict)
                bash_file.close()
        
            elif job_dict["device"] == "MI50_GPU":
                
                bash_file = open(path.join(full_path_out, "TRAIN.sh".format(train_mode)), "w")
                bash_file.write(
"""#!/bin/bash
#SBATCH --job-name=%(job-name)s                                             # Task name
#SBATCH --chdir=%(chdir)s                                                   # Working directory on shared storage
#SBATCH --time=%(time)s                                                     # Run time limit 
#SBATCH --mem=%(mem)s                                                       # job memory
#SBATCH --partition=%(partition)s                                           # job partition (debug, main)
#SBATCH --mail-type=%(mail-type)s                                           # notify via email
#SBATCH --mail-user=%(mail-user)s                                           # recipient
#SBATCH --constraint=mi50   		                                        # reservation for GPU
#SBATCH --gres=gpu:1                                                        # submit to specific GPU

time singularity exec %(rocm_container)s python3 %(job_script)s --train-mode $1 --job-id ${SLURM_JOBID} --local-training-dir $2

""" % job_dict)
                bash_file.close()

            elif job_dict["device"] == "CPU":
                
                bash_file = open(path.join(full_path_out, "TRAIN.sh".format(train_mode)), "w")
                bash_file.write(
"""#!/bin/bash
#SBATCH --job-name=%(job-name)s                                             # Task name
#SBATCH --chdir=%(chdir)s                                                   # Working directory on shared storage
#SBATCH --time=%(time)s                                                     # Run time limit 
#SBATCH --mem=%(mem)s                                                       # job memory
#SBATCH --cpus-per-task=%(cpus-per-task)s                                   # cpus per task
#SBATCH --partition=%(partition)s                                           # job partition (debug, main)
#SBATCH --mail-type=%(mail-type)s                                           # notify via email
#SBATCH --mail-user=%(mail-user)s                                           # recipient

time singularity exec %(cuda_container)s python3 %(job_script)s --train-mode $1 --job-id ${SLURM_JOBID} --local-training-dir $2

""" % job_dict)
                bash_file.close()

            else:
                print("Choose a given device (GPU or CPU)!")
                print("Stopping.")
                exit()
    
    else: ### QA job
 
        actual_job_script = job_script
        
        if job_dict["device"] == "EPN": ### Setup to submit to EPN nodes
            bash_file = open(path.join(full_path_out, "QA.sh"), "w")
            bash_file.write(
   
"""#!/bin/bash
#SBATCH --job-name=%(job-name)s                                                 # Task name
#SBATCH --chdir=%(chdir)s                                                       # Working directory on shared storage
#SBATCH --time=10                                                               # Run time limit 
#SBATCH --mem=30G                                                               # job memory
#SBATCH --cpus-per-task=5                                                       # cpus per task
#SBATCH --partition=prod                                                        # job partition (debug, main)
#SBATCH --mail-type=%(mail-type)s                                               # notify via email
#SBATCH --mail-user=%(mail-user)s                                               # recipient

time python3.9 %(actual_job_script)s --local-training-dir $1

""" % {**job_dict, 'actual_job_script': actual_job_script})

        else:

            bash_file = open(path.join(full_path_out, "QA.sh"), "w")
            bash_file.write(
"""#!/bin/bash
#SBATCH --job-name=%(job-name)s                                                 # Task name
#SBATCH --chdir=%(chdir)s                                                       # Working directory on shared storage
#SBATCH --time=10                                                               # Run time limit 
#SBATCH --mem=30G                                                               # job memory
#SBATCH --cpus-per-task=5                                                       # cpus per task
#SBATCH --partition=main                                                        # job partition (debug, main)
#SBATCH --mail-type=%(mail-type)s                                               # notify via email
#SBATCH --mail-user=%(mail-user)s                                               # recipient

time singularity exec %(cuda_container)s python3 %(actual_job_script)s --local-training-dir $1

""" % {**job_dict, 'actual_job_script': actual_job_script})
            bash_file.close()
        
elif scheduler.lower() == "htcondor":

    bash_file = open(path.join(full_path_out, "TRAIN.sh".format(train_mode)), "w")
    bash_file.write(
"""#!/bin/bash
time python3 %(job_script)s --train-mode $1 --job-id ${CONDOR_JOB_ID} --local-training-dir $2
""" % job_dict)
    bash_file.close()
    os.system("chmod +x {0}".format(path.join(full_path_out, "TRAIN.sh".format(train_mode)))) # For execution on HTCondor
        
    
else:
    print("Scheduler unknown! Check config.json file.")
    exit()