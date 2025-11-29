"""
File: run_job_single_sigma.py
Author: Christian Sonnabend
Email: christian.sonnabend@cern.ch
Date: 15/03/2024
"""

import json, sys, os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="configuration.json", help="Path to the configuration file")
args = parser.parse_args()

### Command line arguments
config = args.config
with open(config, 'r') as config_file:
    CONFIG = json.load(config_file)
sys.path.append(CONFIG['paths']['framework'] + "/framework")
from base import *

LOG = logger.logger(min_severity=CONFIG["process"].get("severity", "DEBUG"), task_name="run_job_single_sigma")

output_folder   = CONFIG["output"]["general"]["training"]
execution_mode  = CONFIG["trainNeuralNetOptions"]["execution_mode"]
training_file   = CONFIG["trainNeuralNetOptions"]["training_file"]
num_networks	= CONFIG["trainNeuralNetOptions"]["num_networks"]
enable_qa		= CONFIG["trainNeuralNetOptions"]["enable_qa"]
scheduler       = CONFIG["trainNeuralNetOptions"]["scheduler"]
qa_dir          = CONFIG["output"]["trainNeuralNet"]["QApath"]


if scheduler == "slurm":

    job_ids = [-1]

    if("RUN12" in execution_mode):
        out = subprocess.check_output("sbatch --output={0}/networks/network_run12/job.out --error={0}/networks/network_run12/job.err {0}/TRAIN.sh {1} RUN12".format(output_folder, args.config), shell=True).decode().strip('\n')
        LOG.info(out)
        job_ids[0] = str(out.split(" ")[-1])

    if ("MEAN" in execution_mode) or (execution_mode=="FULL"):
        ### Submit job for mean calculation
        out = subprocess.check_output("sbatch --output={0}/networks/network_mean/job.out --error={0}/networks/network_mean/job.err {0}/TRAIN.sh {1} MEAN".format(output_folder, args.config), shell=True).decode().strip('\n')
        LOG.info(out)
        job_ids[0] = str(out.split(" ")[-1])

    if ("SIGMA" in execution_mode) or (execution_mode=="FULL"):
        ### Submit job for sigma calculation
        out = subprocess.check_output("sbatch --output={0}/networks/network_sigma/job.out --error={0}/networks/network_sigma/job.err --dependency=afterok:{1} {0}/TRAIN.sh {2} SIGMA".format(output_folder, job_ids[0], args.config), shell=True).decode().strip('\n')
        LOG.info(out)
        job_ids.append(str(out.split(" ")[-1]))

    if execution_mode=="FULL":
        ### Submit job for full network calculation
        out = subprocess.check_output("sbatch --output={0}/networks/network_full/job.out --error={0}/networks/network_full/job.err --dependency=afterok:{1} {0}/TRAIN.sh {2} FULL".format(output_folder, job_ids[-1], args.config), shell=True).decode().strip('\n')
        LOG.info(out)
        job_ids.append(str(out.split(" ")[-1]))

    if enable_qa in ["True", 1]:
        ### Submit job for QA output
        # out = subprocess.check_output("sbatch --output={0}/QA/job.out --error={0}/QA/job.err --dependency=afterok:{1} {0}/QA.sh {2}".format(output_folder, job_ids[-1], args.config), shell=True).decode().strip('\n')
        out = subprocess.check_output("sbatch --output={0}/job.out --error={0}/job.err --dependency=afterok:{1} {0}/QA.sh {2}".format(qa_dir, job_ids[-1], args.config), shell=True).decode().strip('\n')
        LOG.info(out)
        job_ids.append(str(out.split(" ")[-1]))



elif scheduler == "htcondor":

    import htcondor
    import htcondor.dags as dags

    condor_settings = CONFIG["trainNeuralNetOptions"]["htcondor"]

    dag = dags.DAG()
    dag_layers = list()

    # +JobFlavour: espresso = 20 minutes,microcentury = 1 hour,longlunch = 2 hours,workday = 8 hours,tomorrow = 1 day,testmatch = 3 days,nextweek = 1 week

    if ("MEAN" in execution_mode) or (execution_mode=="FULL"):
        ### Submit job for mean calculation
        exec_dict = {
            "executable": output_folder+"/TRAIN.sh",
            "arguments": "MEAN {0}".format(output_folder.split("/")[-1]),
            "output": output_folder+"/networks/network_mean/job.out",       # anything the job prints to standard output will end up in this file
            "error": output_folder+"/networks/network_mean/job.err",        # anything the job prints to standard error will end up in this file
            "log": output_folder+"/networks/network_mean/job.log",          # this file will contain a record of what happened to the job
        }
        condor_settings.update(exec_dict)
        dag_layers.append(dag.layer(name = 'MEAN', submit_description = htcondor.Submit(condor_settings)))

    if ("SIGMA" in execution_mode) or (execution_mode=="FULL"):
        ### Submit job for sigma calculation
        exec_dict = {
            "executable": output_folder+"/TRAIN.sh",
            "+JobFlavour": "workday",
            "arguments": "SIGMA {0}".format(output_folder.split("/")[-1]),
            "output": output_folder+"/networks/network_sigma/job.out",       # anything the job prints to standard output will end up in this file
            "error": output_folder+"/networks/network_sigma/job.err",        # anything the job prints to standard error will end up in this file
            "log": output_folder+"/networks/network_sigma/job.log",          # this file will contain a record of what happened to the job
        }
        condor_settings.update(exec_dict)
        dag_layers.append(dag_layers[-1].child_layer(name = 'SIGMA', submit_description = htcondor.Submit(condor_settings)))

    if execution_mode=="FULL":
        ### Submit job for full network calculation
        exec_dict = {
            "executable": output_folder+"/TRAIN.sh",
            "+JobFlavour": "workday",
            "arguments": "FULL {0}".format(output_folder.split("/")[-1]),
            "output": output_folder+"/networks/network_full/job.out",       # anything the job prints to standard output will end up in this file
            "error": output_folder+"/networks/network_full/job.err",        # anything the job prints to standard error will end up in this file
            "log": output_folder+"/networks/network_full/job.log",          # this file will contain a record of what happened to the job
        }
        condor_settings.update(exec_dict)
        dag_layers.append(dag_layers[-1].child_layer(name = 'FULL', submit_description = htcondor.Submit(condor_settings)))

    dags.write_dag(dag, output_folder)
    dag_submit = htcondor.Submit.from_dag(str(output_folder+"/dagfile.dag"), {'force': 1})

    os.chdir(str(output_folder))
    schedd = htcondor.Schedd()
    cluster_id = schedd.submit(dag_submit).cluster()
    LOG.info(f"DAGMan job cluster is {cluster_id}")
