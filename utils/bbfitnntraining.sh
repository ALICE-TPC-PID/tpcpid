#!/bin/bash

#SBATCH --job-name=BBFittingAndNNTraining
#SBATCH --partition=debug
#SBATCH --array=1
#SBATCH --time=30
#SBATCH --mem=32G   # job memory
#SBATCH --cpus-per-task=2  # cpus per task
#SBATCH --output=BBNNTask_%x-%A_%a.out
#SBATCH --error=BBNNTask_%x-%A_%a.err

#This script should be copied to the job_script_directory by the setup.py and then be submitted as batch job with run.sh
parent_dir="$1"
job_scripts_dir="$2"
sif_path="$1/utils/Container-Build/bbframework/framework_BB_NN.sif"

if [[ -z "$parent_dir" || -z "$job_scripts_dir" ]]; then
	echo "Usage: $0 <parent_dir> <job_scripts_dir>"
	exit 1
fi

apptainer shell -B "${parent_dir}:${parent_dir}" "$sif_path" << SHELLEOF
root -l -b -q 'plotSkimTreeQA2D_modified.C()'
root -l -b -q 'fitNormGraphdEdxvsBGpid_modified.C'
python3 shift_nsigma_modified.py
python3 CreateDataset.py
SHELLEOF
python3 create_jobs.py
python3 run_jobs.py