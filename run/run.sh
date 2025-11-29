#!/bin/bash
#SBATCH --job-name=TPCPID_framework                                                 # Task name
#SBATCH --chdir=/lustre/alice/users/csonnab/TPC/tpcpid-github-official/output/LHC23/pass5/zzh/LHC23zzh_pass5_First_FullTest_TPCSignal_HR_True/20251128/training                                                       # Working directory on shared storage
#SBATCH --time=10                                                               # Run time limit
#SBATCH --mem=30G                                                               # job memory
#SBATCH --cpus-per-task=5                                                       # cpus per task
#SBATCH --partition=debug                                                       # job partition (debug, main)
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND                                               # notify via email
#SBATCH --mail-user=                                               # recipient

python3 run.py --config $1