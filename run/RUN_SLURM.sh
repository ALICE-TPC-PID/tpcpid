#!/bin/bash
#SBATCH --job-name=TPCPID_MASTERJOB                                         # Task name
#SBATCH --chdir=/lustre/alice/users/csonnab/TPC/tpcpid-github-official                                                         # Working directory on shared storage
#SBATCH --time=10                                                           # Run time limit
#SBATCH --mem=30G                                                           # job memory
#SBATCH --partition=debug                                                   # job partition (debug, main)
#SBATCH --output=/lustre/alice/users/csonnab/TPC/tpcpid-github-official/output/LHC23/zzf/apass5/LHC23zzf_apass5_TPCSignal_HadronicRate/20251130/run_%j.out                                             # Standard output and error log
#SBATCH --error=/lustre/alice/users/csonnab/TPC/tpcpid-github-official/output/LHC23/zzf/apass5/LHC23zzf_apass5_TPCSignal_HadronicRate/20251130/run_%j.err                                              # Standard error log

time python3 /lustre/alice/users/csonnab/TPC/tpcpid-github-official/run/src/run_framework.py --config $1
