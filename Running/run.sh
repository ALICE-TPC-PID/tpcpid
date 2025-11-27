#!/bin/bash

#This script will be executed via bash run.sh in the Running dir.
#It will run the setup script, cd into the job_script dir, and submit the full BBandNN process there

python3 setup.py
parent_dir=$(python3 -c 'import json; print(json.load(open("configuration.json"))["output"]["general"]["base_folder"])')
job_script_dir=$(python3 -c 'import json; print(json.load(open("configuration.json"))["output"]["general"]["job"])')

echo "$parent_dir"
echo "$job_script_dir"

cd $job_script_dir
echo "$PWD"
sbatch bbfitnntraining.sh ${parent_dir} ${job_script_dir}