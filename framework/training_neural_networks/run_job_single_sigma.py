"""
File: run_job_single_sigma.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
"""

import json
import sys
import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="configuration.json", help="Path to the configuration file")
parser.add_argument("-ci", "--ci-run", type=int, default=0, help="Run in CI mode (stream output to terminal)")
args = parser.parse_args()

### Command line arguments
config = args.config
with open(config, 'r') as config_file:
    CONFIG = json.load(config_file)

sys.path.append(CONFIG['settings']['framework'] + "/framework")
from base import *

LOG = logger(min_severity=CONFIG["process"].get("severity", "DEBUG"), task_name="run_job_single_sigma")

output_folder   = CONFIG["output"]["general"]["training"]
execution_mode  = CONFIG["trainNeuralNetOptions"]["execution_mode"]
training_file   = CONFIG["trainNeuralNetOptions"]["training_file"]
num_networks    = CONFIG["trainNeuralNetOptions"]["num_networks"]
enable_qa       = CONFIG["trainNeuralNetOptions"]["enable_qa"]
scheduler       = CONFIG["trainNeuralNetOptions"]["scheduler"]
qa_dir          = CONFIG["output"]["trainNeuralNet"]["QApath"]


def ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_local_step(name, cmd, stdout_path, stderr_path, cwd=None):
    ensure_parent_dir(stdout_path)
    ensure_parent_dir(stderr_path)

    LOG.info(f"Starting local step: {name}")
    LOG.info("Command: " + " ".join(cmd))

    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        proc = subprocess.run(
            cmd,
            stdout=fout,
            stderr=ferr,
            cwd=cwd,
            check=False
        )

    if proc.returncode != 0:
        LOG.fatal(f"Step {name} failed with return code {proc.returncode}")

    LOG.info(f"Step {name} finished successfully")


if scheduler == "slurm":

    job_ids = [-1]

    if ("RUN12" in execution_mode):
        out = subprocess.check_output(
            "sbatch --output={0}/networks/network_run12/job.out "
            "--error={0}/networks/network_run12/job.err "
            "{0}/TRAIN.sh {1} RUN12".format(output_folder, args.config),
            shell=True
        ).decode().strip('\n')
        LOG.info(out)
        job_ids[0] = str(out.split(" ")[-1])

    if ("MEAN" in execution_mode) or (execution_mode == "FULL"):
        ### Submit job for mean calculation
        out = subprocess.check_output(
            "sbatch --output={0}/networks/network_mean/job.out "
            "--error={0}/networks/network_mean/job.err "
            "{0}/TRAIN.sh {1} MEAN".format(output_folder, args.config),
            shell=True
        ).decode().strip('\n')
        LOG.info(out)
        job_ids[0] = str(out.split(" ")[-1])

    if ("SIGMA" in execution_mode) or (execution_mode == "FULL"):
        ### Submit job for sigma calculation
        out = subprocess.check_output(
            "sbatch --output={0}/networks/network_sigma/job.out "
            "--error={0}/networks/network_sigma/job.err "
            "--dependency=afterok:{1} {0}/TRAIN.sh {2} SIGMA".format(output_folder, job_ids[0], args.config),
            shell=True
        ).decode().strip('\n')
        LOG.info(out)
        job_ids.append(str(out.split(" ")[-1]))

    if execution_mode == "FULL":
        ### Submit job for full network calculation
        out = subprocess.check_output(
            "sbatch --output={0}/networks/network_full/job.out "
            "--error={0}/networks/network_full/job.err "
            "--dependency=afterok:{1} {0}/TRAIN.sh {2} FULL".format(output_folder, job_ids[-1], args.config),
            shell=True
        ).decode().strip('\n')
        LOG.info(out)
        job_ids.append(str(out.split(" ")[-1]))

    if enable_qa in ["True", 1, True]:
        ### Submit job for QA output
        out = subprocess.check_output(
            "sbatch --output={0}/job.out --error={0}/job.err "
            "--dependency=afterok:{1} {0}/QA.sh {2}".format(qa_dir, job_ids[-1], args.config),
            shell=True
        ).decode().strip('\n')
        LOG.info(out)
        job_ids.append(str(out.split(" ")[-1]))


elif scheduler == "htcondor":

    import htcondor
    import htcondor.dags as dags

    condor_settings = CONFIG["trainNeuralNetOptions"]["htcondor"]

    dag = dags.DAG()
    dag_layers = list()

    # +JobFlavour: espresso = 20 minutes,microcentury = 1 hour,longlunch = 2 hours,workday = 8 hours,tomorrow = 1 day,testmatch = 3 days,nextweek = 1 week

    if ("MEAN" in execution_mode) or (execution_mode == "FULL"):
        ### Submit job for mean calculation
        exec_dict = {
            "executable": output_folder + "/TRAIN.sh",
            "arguments": "{0} MEAN".format(args.config),
            "output": output_folder + "/networks/network_mean/job.out",
            "error": output_folder + "/networks/network_mean/job.err",
            "log": output_folder + "/networks/network_mean/job.log",
        }
        condor_settings.update(exec_dict)
        dag_layers.append(dag.layer(name='MEAN', submit_description=htcondor.Submit(condor_settings)))

    if ("SIGMA" in execution_mode) or (execution_mode == "FULL"):
        ### Submit job for sigma calculation
        exec_dict = {
            "executable": output_folder + "/TRAIN.sh",
            "+JobFlavour": "workday",
            "arguments": "{0} SIGMA".format(args.config),
            "output": output_folder + "/networks/network_sigma/job.out",
            "error": output_folder + "/networks/network_sigma/job.err",
            "log": output_folder + "/networks/network_sigma/job.log",
        }
        condor_settings.update(exec_dict)
        dag_layers.append(dag_layers[-1].child_layer(name='SIGMA', submit_description=htcondor.Submit(condor_settings)))

    if execution_mode == "FULL":
        ### Submit job for full network calculation
        exec_dict = {
            "executable": output_folder + "/TRAIN.sh",
            "+JobFlavour": "workday",
            "arguments": "{0} FULL".format(args.config),
            "output": output_folder + "/networks/network_full/job.out",
            "error": output_folder + "/networks/network_full/job.err",
            "log": output_folder + "/networks/network_full/job.log",
        }
        condor_settings.update(exec_dict)
        dag_layers.append(dag_layers[-1].child_layer(name='FULL', submit_description=htcondor.Submit(condor_settings)))

    dags.write_dag(dag, output_folder)
    dag_submit = htcondor.Submit.from_dag(str(output_folder + "/dagfile.dag"), {'force': 1})

    os.chdir(str(output_folder))
    schedd = htcondor.Schedd()
    cluster_id = schedd.submit(dag_submit).cluster()
    LOG.info(f"DAGMan job cluster is {cluster_id}")


elif scheduler == "local":

    train_sh = os.path.join(output_folder, "TRAIN.sh")
    qa_sh = os.path.join(qa_dir, "QA.sh")

    if not os.path.isfile(train_sh):
        LOG.fatal(f"Missing training script: {train_sh}")

    steps = []

    if "RUN12" in execution_mode:
        steps.append({
            "name": "RUN12",
            "cmd": [train_sh, args.config, "RUN12"],
            "stdout": os.path.join(output_folder, "networks/network_run12/job.out"),
            "stderr": os.path.join(output_folder, "networks/network_run12/job.err"),
            "cwd": output_folder,
        })

    if ("MEAN" in execution_mode) or (execution_mode == "FULL"):
        steps.append({
            "name": "MEAN",
            "cmd": [train_sh, args.config, "MEAN"],
            "stdout": os.path.join(output_folder, "networks/network_mean/job.out"),
            "stderr": os.path.join(output_folder, "networks/network_mean/job.err"),
            "cwd": output_folder,
        })

    if ("SIGMA" in execution_mode) or (execution_mode == "FULL"):
        steps.append({
            "name": "SIGMA",
            "cmd": [train_sh, args.config, "SIGMA"],
            "stdout": os.path.join(output_folder, "networks/network_sigma/job.out"),
            "stderr": os.path.join(output_folder, "networks/network_sigma/job.err"),
            "cwd": output_folder,
        })

    if execution_mode == "FULL":
        steps.append({
            "name": "FULL",
            "cmd": [train_sh, args.config, "FULL"],
            "stdout": os.path.join(output_folder, "networks/network_full/job.out"),
            "stderr": os.path.join(output_folder, "networks/network_full/job.err"),
            "cwd": output_folder,
        })

    if enable_qa in ["True", 1, True]:
        if not os.path.isfile(qa_sh):
            LOG.info(f"Missing QA script: {qa_sh}")
            LOG.info("Stopping.")
            sys.exit(1)

        steps.append({
            "name": "QA",
            "cmd": [qa_sh, args.config],
            "stdout": os.path.join(qa_dir, "job.out"),
            "stderr": os.path.join(qa_dir, "job.err"),
            "cwd": qa_dir,
        })

    if len(steps) == 0:
        LOG.info("No local steps selected from execution_mode.")
        sys.exit(0)

    for step in steps:
        LOG.info(f"Starting step: {step['name']}")
        LOG.info("Command: " + " ".join(step["cmd"]))

        if getattr(args, "ci_run", False):
            proc = subprocess.run(
                step["cmd"],
                cwd=step["cwd"],
                stderr=subprocess.STDOUT  # merge stderr into stdout
            )
        else:
            ensure_parent_dir(step["stdout"])
            ensure_parent_dir(step["stderr"])

            with open(step["stdout"], "w") as fout, open(step["stderr"], "w") as ferr:
                proc = subprocess.run(
                    step["cmd"],
                    stdout=fout,
                    stderr=ferr,
                    cwd=step["cwd"],
                )

        if proc.returncode != 0:
            LOG.error(f"Step {step['name']} failed with return code {proc.returncode}")
            sys.exit(proc.returncode)

        LOG.info(f"Step {step['name']} finished successfully")

    LOG.info("All local steps finished successfully")


else:
    LOG.info("Scheduler unknown! Check config.json file.")
    sys.exit(1)