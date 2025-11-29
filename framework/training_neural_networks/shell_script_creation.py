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

from os import path

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="configuration.json", help="Path to the configuration file")
parser.add_argument("-jbscript", "--job-script", default=".", help="Path to job script")
parser.add_argument("-trm", "--training-mode", default='MEAN', help="Training mode") #  choices=['MEAN', 'SIGMA', 'FULL', 'ENSEMBLE'] fails (?)
args = parser.parse_args()

job_script      = str(args.job_script)
train_mode      = str(args.training_mode)

with open(args.config, 'r') as config_file:
    CONFIG = json.load(config_file)
sys.path.append(CONFIG['paths']['framework'] + "/framework")
from base import *

LOG = logger.logger(min_severity=CONFIG["process"].get("severity", "DEBUG"), task_name="shell_script_creation")

output_folder           = CONFIG["output"]["general"]["training"]
scheduler               = CONFIG["trainNeuralNetOptions"]["scheduler"]
job_dict                = CONFIG["trainNeuralNetOptions"][scheduler]

full_path_out           = output_folder
qa_dir                  = CONFIG["output"]["trainNeuralNet"]["QApath"]
job_dict["chdir"]       = full_path_out
job_dict["job_script"]  = job_script

if scheduler.lower() == "slurm":

    if train_mode != "QA":

        if job_dict["device"] == "EPN": ### Setup to submit to EPN nodes

            bash_path = path.join(full_path_out, "TRAIN.sh")
            script = """#!/bin/bash
#SBATCH --job-name=%(name)s                                                 # Task name
#SBATCH --chdir=%(pj)s                                                      # Working directory on shared storage
#SBATCH --time=%(time)s                                                     # Run time limit
#SBATCH --mem=%(mem)s                                                       # job memory
#SBATCH --partition=%(part)s                                                # job partition (debug, main)
#SBATCH --mail-type=%(notify)s                                              # notify via email
#SBATCH --mail-user=%(email)s                                               # recipient
""" % job_dict

            if "ngpus" in job_dict.keys() and int(job_dict['ngpus']) > 8:
                job_dict['nodes'] = int(job_dict['ngpus']) // 8
                job_dict['ntasks_per_node'] = 8
                script += """#SBATCH --nodes=%(nodes)s                      # number of nodes
#SBATCH --gres=gpu:8   		                                                # reservation for GPU
#SBATCH --ntasks-per-node=%(ntasks_per_node)s                               # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time srun /bin/python3.9 python3 %(job_script)s --config $1 --train-mode $2
""" % job_dict

            else:
                script += """#SBATCH --nodes=1                              # number of nodes
#SBATCH --gres=gpu:%(ngpus)s   		                                        # reservation for GPU
#SBATCH --ntasks-per-node=%(ngpus)s                                         # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time srun /bin/python3.9 python3 %(job_script)s --config $1 --train-mode $2
""" % job_dict

            bash_file = open(bash_path, "w")
            bash_file.write(script)
            bash_file.close()

        else: ### Setup for GSI batch farm (default)

            if job_dict["device"] == "MI100_GPU":

                bash_path = path.join(full_path_out, "TRAIN.sh")
                script="""#!/bin/bash
#SBATCH --job-name=%(job-name)s                                             # Task name
#SBATCH --chdir=%(chdir)s                                                   # Working directory on shared storage
#SBATCH --time=%(time)s                                                     # Run time limit
#SBATCH --mem=%(mem)s                                                       # job memory
#SBATCH --partition=gpu                                                     # job partition (debug, main)
#SBATCH --mail-type=%(mail-type)s                                           # notify via email
#SBATCH --mail-user=%(mail-user)s                                           # recipient
#SBATCH --constraint=mi100
""" % job_dict
                if "ngpus" in job_dict.keys() and int(job_dict['ngpus']) > 8:
                    job_dict['nodes'] = int(job_dict['ngpus']) // 8
                    job_dict['ntasks_per_node'] = 8
                    script += """#SBATCH --nodes=%(nodes)s                  # number of nodes
#SBATCH --gres=gpu:8   		                                                # reservation for GPU
#SBATCH --ntasks-per-node=%(ntasks_per_node)s                               # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time srun singularity exec %(rocm_container)s python3 %(job_script)s --config $1 --train-mode $2
""" % job_dict

                else:
                    script += """#SBATCH --nodes=1                          # number of nodes
#SBATCH --gres=gpu:%(ngpus)s   		                                        # reservation for GPU
#SBATCH --ntasks-per-node=%(ngpus)s                                         # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time srun singularity exec %(rocm_container)s python3 %(job_script)s --config $1 --train-mode $2
""" % job_dict

                bash_file = open(bash_path, "w")
                bash_file.write(script)
                bash_file.close()

            elif job_dict["device"] == "MI50_GPU":

                bash_path = path.join(full_path_out, "TRAIN.sh")
                script="""#!/bin/bash
#SBATCH --job-name=%(job-name)s                                             # Task name
#SBATCH --chdir=%(chdir)s                                                   # Working directory on shared storage
#SBATCH --time=%(time)s                                                     # Run time limit
#SBATCH --mem=%(mem)s                                                       # job memory
#SBATCH --partition=gpu                                                     # job partition (debug, main)
#SBATCH --mail-type=%(mail-type)s                                           # notify via email
#SBATCH --mail-user=%(mail-user)s                                           # recipient
#SBATCH --constraint=mi50
""" % job_dict
                if "ngpus" in job_dict.keys() and int(job_dict['ngpus']) > 8:
                    job_dict['nodes'] = int(job_dict['ngpus']) // 8
                    job_dict['ntasks_per_node'] = 8
                    script += """#SBATCH --nodes=1                          # number of nodes
#SBATCH --gres=gpu:8   		                                                # reservation for GPU
#SBATCH --ntasks-per-node=%(ntasks_per_node)s                               # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time srun singularity exec %(rocm_container)s python3 %(job_script)s --config $1 --train-mode $2
""" % job_dict

                else:
                    script += """#SBATCH --nodes=1                          # number of nodes
#SBATCH --gres=gpu:%(ngpus)s   		                                        # reservation for GPU
#SBATCH --ntasks-per-node=%(ngpus)s                                         # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time srun singularity exec %(rocm_container)s python3 %(job_script)s --config $1 --train-mode $2
""" % job_dict

                bash_file = open(bash_path, "w")
                bash_file.write(script)
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

time singularity exec %(cuda_container)s python3 %(job_script)s --config $1 --train-mode $2

""" % job_dict)
                bash_file.close()

            else:
                LOG.info("Choose a given device (GPU or CPU)!")
                LOG.info("Stopping.")
                exit()

    else: ### QA job

        actual_job_script = job_script

        if job_dict["device"] == "EPN": ### Setup to submit to EPN nodes
            bash_file = open(path.join(qa_dir, "QA.sh"), "w")
            bash_file.write(

"""#!/bin/bash
#SBATCH --job-name=%(job-name)s                                                 # Task name
#SBATCH --chdir=%(chdir)s                                                       # Working directory on shared storage
#SBATCH --time=10                                                               # Run time limit
#SBATCH --mem=30G                                                               # job memory
#SBATCH --partition=prod                                                        # job partition (debug, main)
#SBATCH --mail-type=%(mail-type)s                                               # notify via email
#SBATCH --mail-user=%(mail-user)s                                               # recipient

time python3.9 %(actual_job_script)s --config $1

""" % {**job_dict, 'actual_job_script': actual_job_script})

        else:

            bash_file = open(path.join(qa_dir, "QA.sh"), "w")
            bash_file.write(
"""#!/bin/bash
#SBATCH --job-name=%(job-name)s                                                 # Task name
#SBATCH --chdir=%(chdir)s                                                       # Working directory on shared storage
#SBATCH --time=10                                                               # Run time limit
#SBATCH --mem=30G                                                               # job memory
#SBATCH --partition=debug                                                       # job partition (debug, main)
#SBATCH --mail-type=%(mail-type)s                                               # notify via email
#SBATCH --mail-user=%(mail-user)s                                               # recipient

time singularity exec %(cuda_container)s python3 %(actual_job_script)s --config $1

""" % {**job_dict, 'actual_job_script': actual_job_script})
            bash_file.close()

elif scheduler.lower() == "htcondor":

    bash_file = open(path.join(full_path_out, "TRAIN.sh".format(train_mode)), "w")
    bash_file.write(
"""#!/bin/bash
time python3 %(job_script)s --config $1 --train-mode $2
""" % job_dict)
    bash_file.close()
    os.system("chmod +x {0}".format(path.join(full_path_out, "TRAIN.sh".format(train_mode)))) # For execution on HTCondor


else:
    LOG.info("Scheduler unknown! Check config.json file.")
    exit()