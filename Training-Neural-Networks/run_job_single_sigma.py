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

### Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-cd", "--current-dir", default=".", help="Directory for training of the neural network")
parser.add_argument("-sched", "--scheduler", default="slurm", help="System for job-scheduling")
parser.add_argument("-config", "--config", default="./config.json", help="Configurations file")
args = parser.parse_args()

current_dir = str(args.current_dir)

### External json settings
configs_file = open(args.config, "r")
CONF = json.load(configs_file)

### directory settings
training_dir    = CONF["directories"]["training_dir"]
output_folder   = CONF["directories"]["output_folder"]

### network settings
execution_mode  = CONF["network"]["execution_mode"]
training_file   = CONF["network"]["training_file"]
num_networks	= CONF["network"]["num_networks"]
enable_qa		= CONF["network"]["enable_qa"]

configs_file.close()

if args.scheduler == "slurm":

    job_ids = [-1]
    optional_args   = CONF["submission"]["slurm"]["optional_args"]

    if("RUN12" in execution_mode):
        out = subprocess.check_output("sbatch --output={0}/networks/network_run12/job.out --error={0}/networks/network_run12/job.err {3} {1}/{2}/TRAIN.sh RUN12 {0}".format(current_dir.split("/")[-1], training_dir, output_folder, optional_args), shell=True).decode().strip('\n')
        print(out)
        job_ids[0] = str(out.split(" ")[-1])
 
    if ("MEAN" in execution_mode) or (execution_mode=="FULL"):
        ### Submit job for mean calculation
        out = subprocess.check_output("sbatch --output={0}/networks/network_mean/job.out --error={0}/networks/network_mean/job.err {3} {1}/{2}/TRAIN.sh MEAN {0}".format(current_dir.split("/")[-1], training_dir, output_folder, optional_args), shell=True).decode().strip('\n')
        print(out)
        job_ids[0] = str(out.split(" ")[-1])

    if ("SIGMA" in execution_mode) or (execution_mode=="FULL"):
        ### Submit job for sigma calculation
        out = subprocess.check_output("sbatch --output={0}/networks/network_sigma/job.out --error={0}/networks/network_sigma/job.err {4} --dependency=afterok:{1} {2}/{3}/TRAIN.sh SIGMA {0}".format(current_dir.split("/")[-1], job_ids[-1], training_dir, output_folder, optional_args), shell=True).decode().strip('\n')
        print(out)
        job_ids.append(str(out.split(" ")[-1]))

    if execution_mode=="FULL":
        ### Submit job for full network calculation
        out = subprocess.check_output("sbatch --output={0}/networks/network_full/job.out --error={0}/networks/network_full/job.err {4} --dependency=afterok:{1} {2}/{3}/TRAIN.sh FULL {0}".format(current_dir.split("/")[-1], job_ids[-1], training_dir, output_folder, optional_args), shell=True).decode().strip('\n')
        print(out)
        job_ids.append(str(out.split(" ")[-1]))
    
    if enable_qa in ["True", 1]:
        ### Submit job for QA output
        out = subprocess.check_output("sbatch --output={0}/QA/job.out --error={0}/QA/job.err --dependency=afterok:{1} {2}/{3}/QA.sh {0}".format(training_dir + "/" + output_folder + "/" + current_dir.split("/")[-1], job_ids[-1], training_dir, output_folder), shell=True).decode().strip('\n')
        print(out)
        job_ids.append(str(out.split(" ")[-1]))
        

    # if "ENSEMBLE" in execution_mode:
    # 	os.system("python3 {0}/shell_script_creation.py {1} ENSEMBLE {2}".format(training_dir+"/configurations", training_dir+"/"+output_folder+"/train.py", -1))
    # 	for i in range(num_networks):
    # 		out = subprocess.check_output("sbatch --output={2}/networks/job.out --error={2}/networks/job.err {0}/{1}/TRAIN.sh {2}".format(training_dir, output_folder, current_dir.split("/")[-1]), shell=True).decode().strip('\n')
    # 		print(out)

elif args.scheduler == "htcondor":

    import htcondor
    import htcondor.dags as dags
    
    condor_settings = CONF["submission"]["htcondor"]
    
    dag = dags.DAG()
    dag_layers = list()

    # +JobFlavour: espresso = 20 minutes,microcentury = 1 hour,longlunch = 2 hours,workday = 8 hours,tomorrow = 1 day,testmatch = 3 days,nextweek = 1 week
    
    if ("MEAN" in execution_mode) or (execution_mode=="FULL"):
        ### Submit job for mean calculation
        exec_dict = {
            "executable": training_dir+"/"+output_folder+"/TRAIN.sh",
            "arguments": "MEAN {0}".format(current_dir.split("/")[-1]),
            "output": training_dir+"/"+output_folder+"/"+current_dir.split("/")[-1]+"/networks/network_mean/job.out",       # anything the job prints to standard output will end up in this file
            "error": training_dir+"/"+output_folder+"/"+current_dir.split("/")[-1]+"/networks/network_mean/job.err",        # anything the job prints to standard error will end up in this file
            "log": training_dir+"/"+output_folder+"/"+current_dir.split("/")[-1]+"/networks/network_mean/job.log",          # this file will contain a record of what happened to the job
        }
        condor_settings.update(exec_dict)
        dag_layers.append(dag.layer(name = 'MEAN', submit_description = htcondor.Submit(condor_settings)))

    if ("SIGMA" in execution_mode) or (execution_mode=="FULL"):
        ### Submit job for sigma calculation
        exec_dict = {
            "executable": training_dir+"/"+output_folder+"/TRAIN.sh",
            "+JobFlavour": "workday",
            "arguments": "SIGMA {0}".format(current_dir.split("/")[-1]),
            "output": training_dir+"/"+output_folder+"/"+current_dir.split("/")[-1]+"/networks/network_sigma/job.out",       # anything the job prints to standard output will end up in this file
            "error": training_dir+"/"+output_folder+"/"+current_dir.split("/")[-1]+"/networks/network_sigma/job.err",        # anything the job prints to standard error will end up in this file
            "log": training_dir+"/"+output_folder+"/"+current_dir.split("/")[-1]+"/networks/network_sigma/job.log",          # this file will contain a record of what happened to the job
        }
        condor_settings.update(exec_dict)
        dag_layers.append(dag_layers[-1].child_layer(name = 'SIGMA', submit_description = htcondor.Submit(condor_settings)))

    if execution_mode=="FULL":
        ### Submit job for full network calculation
        exec_dict = {
            "executable": training_dir+"/"+output_folder+"/TRAIN.sh",
            "+JobFlavour": "workday",
            "arguments": "FULL {0}".format(current_dir.split("/")[-1]),
            "output": training_dir+"/"+output_folder+"/"+current_dir.split("/")[-1]+"/networks/network_full/job.out",       # anything the job prints to standard output will end up in this file
            "error": training_dir+"/"+output_folder+"/"+current_dir.split("/")[-1]+"/networks/network_full/job.err",        # anything the job prints to standard error will end up in this file
            "log": training_dir+"/"+output_folder+"/"+current_dir.split("/")[-1]+"/networks/network_full/job.log",          # this file will contain a record of what happened to the job
        }
        condor_settings.update(exec_dict)
        dag_layers.append(dag_layers[-1].child_layer(name = 'FULL', submit_description = htcondor.Submit(condor_settings)))

    dags.write_dag(dag, training_dir+"/"+output_folder)
    dag_submit = htcondor.Submit.from_dag(str(training_dir+"/"+output_folder+"/dagfile.dag"), {'force': 1})
    
    os.chdir(str(training_dir+"/"+output_folder))
    schedd = htcondor.Schedd()
    cluster_id = schedd.submit(dag_submit).cluster()
    print(f"DAGMan job cluster is {cluster_id}")
