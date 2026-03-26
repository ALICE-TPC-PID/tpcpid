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
parser.add_argument("-trm", "--training-mode", default='MEAN', help="Training mode")
args = parser.parse_args()

job_script = str(args.job_script)
train_mode = str(args.training_mode)

with open(args.config, 'r') as config_file:
    CONFIG = json.load(config_file)

sys.path.append(CONFIG['settings']['framework'] + "/framework")
from base import *

LOG = logger(min_severity=CONFIG["process"].get("severity", "DEBUG"), task_name="shell_script_creation")

output_folder = CONFIG["output"]["general"]["training"]
scheduler = CONFIG["trainNeuralNetOptions"]["scheduler"]
qa_dir = CONFIG["output"]["trainNeuralNet"]["QApath"]

if scheduler not in CONFIG["trainNeuralNetOptions"]:
    LOG.info(f"Scheduler '{scheduler}' not found in config.")
    LOG.info("Stopping.")
    exit()

job_dict = dict(CONFIG["trainNeuralNetOptions"][scheduler])
full_path_out = output_folder
job_dict["chdir"] = full_path_out
job_dict["job_script"] = job_script


def write_script(script_path, script_content, make_executable=True):
    with open(script_path, "w") as bash_file:
        bash_file.write(script_content)
    if make_executable:
        os.chmod(script_path, 0o755)


def get_exec_command(job_dict):
    use_container = job_dict.get("use_container", False)
    device = job_dict.get("device", "CPU")
    python_cmd = job_dict.get("python", "python3")

    if not use_container:
        return python_cmd

    runtime = job_dict.get("container_runtime", "apptainer").lower()

    if runtime != "apptainer":
        raise ValueError(f"Unsupported container runtime: {runtime}")

    if device == "HYDRA":
        return f'apptainer exec --nv "{job_dict["hydra_container"]}" {python_cmd}'
    elif device == "MI100_GPU":
        return f'apptainer exec "{job_dict["rocm_container"]}" {python_cmd}'
    elif device == "CPU" or scheduler.lower() == "local":
        return f'apptainer exec "{job_dict["cuda_container"]}" {python_cmd}'
    elif device == "EPN":
        return job_dict.get("python", "python3.9")
    else:
        raise ValueError(f"Unknown device: {device}")


if scheduler.lower() == "slurm":

    if train_mode != "QA":

        exec_cmd = get_exec_command(job_dict)
        bash_path = path.join(full_path_out, "TRAIN.sh")

        if job_dict["device"] == "EPN":
            script = """#!/bin/bash
#SBATCH --job-name=%(name)s
#SBATCH --chdir=%(pj)s
#SBATCH --time=%(time)s
#SBATCH --mem=%(mem)s
#SBATCH --partition=%(part)s
#SBATCH --mail-type=%(notify)s
#SBATCH --mail-user=%(email)s
""" % job_dict

            if "ngpus" in job_dict and int(job_dict["ngpus"]) > 8:
                job_dict["nodes"] = int(job_dict["ngpus"]) // 8
                job_dict["ntasks_per_node"] = 8
                script += f"""#SBATCH --nodes={job_dict['nodes']}
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node={job_dict['ntasks_per_node']}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}

time srun {exec_cmd} "{job_script}" --config "$1" --train-mode "$2"
"""
            else:
                script += f"""#SBATCH --nodes=1
#SBATCH --gres=gpu:{job_dict['ngpus']}
#SBATCH --ntasks-per-node={job_dict['ngpus']}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}

time srun {exec_cmd} "{job_script}" --config "$1" --train-mode "$2"
"""
            write_script(bash_path, script)

        elif job_dict["device"] == "MI100_GPU":
            script = """#!/bin/bash
#SBATCH --job-name=%(job-name)s
#SBATCH --chdir=%(chdir)s
#SBATCH --time=%(time)s
#SBATCH --mem=%(mem)s
#SBATCH --partition=gpu
#SBATCH --mail-type=%(mail-type)s
#SBATCH --mail-user=%(mail-user)s
#SBATCH --constraint=mi100
""" % job_dict

            if "ngpus" in job_dict and int(job_dict["ngpus"]) > 8:
                job_dict["nodes"] = int(job_dict["ngpus"]) // 8
                job_dict["ntasks_per_node"] = 8
                script += f"""#SBATCH --nodes={job_dict['nodes']}
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node={job_dict['ntasks_per_node']}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}

time srun {exec_cmd} "{job_script}" --config "$1" --train-mode "$2"
"""
            else:
                script += f"""#SBATCH --nodes=1
#SBATCH --gres=gpu:{job_dict['ngpus']}
#SBATCH --ntasks-per-node={job_dict['ngpus']}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}

time srun {exec_cmd} "{job_script}" --config "$1" --train-mode "$2"
"""
            write_script(bash_path, script)

        elif job_dict["device"] == "HYDRA":
            script = """#!/bin/bash
#SBATCH --job-name=%(job-name)s
#SBATCH --chdir=%(chdir)s
#SBATCH --time=%(time)s
#SBATCH --mem=%(mem)s
#SBATCH --partition=gpu
#SBATCH --mail-type=%(mail-type)s
#SBATCH --mail-user=%(mail-user)s
""" % job_dict

            if "ngpus" in job_dict and int(job_dict["ngpus"]) > 8:
                job_dict["nodes"] = int(job_dict["ngpus"]) // 8
                job_dict["ntasks_per_node"] = 8
                script += f"""#SBATCH --nodes={job_dict['nodes']}
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node={job_dict['ntasks_per_node']}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}

time srun {exec_cmd} "{job_script}" --config "$1" --train-mode "$2"
"""
            else:
                script += f"""#SBATCH --nodes=1
#SBATCH --gres=gpu:{job_dict['ngpus']}
#SBATCH --ntasks-per-node={job_dict['ngpus']}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}

time srun {exec_cmd} "{job_script}" --config "$1" --train-mode "$2"
"""
            write_script(bash_path, script)

        elif job_dict["device"] == "CPU":
            script = f"""#!/bin/bash
#SBATCH --job-name={job_dict['job-name']}
#SBATCH --chdir={job_dict['chdir']}
#SBATCH --time={job_dict['time']}
#SBATCH --mem={job_dict['mem']}
#SBATCH --cpus-per-task={job_dict['cpus-per-task']}
#SBATCH --partition={job_dict['partition']}
#SBATCH --mail-type={job_dict['mail-type']}
#SBATCH --mail-user={job_dict['mail-user']}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}

time {exec_cmd} "{job_script}" --config "$1" --train-mode "$2"
"""
            write_script(bash_path, script)

        else:
            LOG.info("Choose a given device (CPU, MI100_GPU, HYDRA, EPN)!")
            LOG.info("Stopping.")
            exit()

    else:
        exec_cmd = get_exec_command(job_dict)
        bash_path = path.join(qa_dir, "QA.sh")

        if job_dict["device"] == "EPN":
            script = f"""#!/bin/bash
#SBATCH --job-name=TPCPID_NNQA
#SBATCH --chdir={qa_dir}
#SBATCH --time=10
#SBATCH --mem=30G
#SBATCH --partition=prod
#SBATCH --mail-type={job_dict['mail-type']}
#SBATCH --mail-user={job_dict['mail-user']}

time {exec_cmd} "{job_script}" --config "$1"
"""
        else:
            script = f"""#!/bin/bash
#SBATCH --job-name=TPCPID_NNQA
#SBATCH --chdir={qa_dir}
#SBATCH --time=10
#SBATCH --mem=30G
#SBATCH --partition=debug
#SBATCH --mail-type={job_dict['mail-type']}
#SBATCH --mail-user={job_dict['mail-user']}

time {exec_cmd} "{job_script}" --config "$1"
"""
        write_script(bash_path, script)


elif scheduler.lower() == "htcondor":

    bash_path = path.join(full_path_out, "TRAIN.sh")
    script = f"""#!/bin/bash
time python3 "{job_script}" --config "$1" --train-mode "$2"
"""
    write_script(bash_path, script)


elif scheduler.lower() == "local":

    try:
        local_exec = get_exec_command(job_dict)
    except Exception as e:
        LOG.info(str(e))
        LOG.info("Stopping.")
        exit()

    if train_mode != "QA":
        bash_path = path.join(full_path_out, "TRAIN.sh")
        script = f"""#!/bin/bash
set -euo pipefail

cd "{full_path_out}"

export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-{job_dict.get('cpus-per-task', 1)}}}"

time {local_exec} "{job_script}" --config "$1" --train-mode "$2"
"""
        write_script(bash_path, script)
        LOG.info(f"Created local training script: {bash_path}")

    else:
        bash_path = path.join(qa_dir, "QA.sh")
        script = f"""#!/bin/bash
set -euo pipefail

cd "{qa_dir}"

export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-{job_dict.get('cpus-per-task', 1)}}}"

time {local_exec} "{job_script}" --config "$1"
"""
        write_script(bash_path, script)
        LOG.info(f"Created local QA script: {bash_path}")


else:
    LOG.info("Scheduler unknown! Check config.json file.")
    exit()