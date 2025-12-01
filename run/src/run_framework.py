import os, sys, json, subprocess
from argparse import ArgumentParser

parser = ArgumentParser(description="Setup the TPC PID analysis")
parser.add_argument("-c", "--config", type=str, default="configuration.json", help="Path to configuration file")
args = parser.parse_args()

with open(args.config, 'r') as config_file:
    CONFIG = json.load(config_file)
sys.path.append(CONFIG['settings']['framework'] + "/framework")
from base import *

try:
    LOG = logger.logger("Framework")
    LOG.framework("Setup completed successfully. Ready to launch!")
    
    full_git_config(
        save_to_file=os.path.join(CONFIG["output"]["general"]["path"], "git_info.txt"),
        verbose=False,
        path=CONFIG['settings']['framework']
    )
    
    LOG.framework("--- Starting plotSkimTreeQA2D_modified.C ---")

    subprocess.run([
        "singularity", "exec",
        "/lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif",
        "root", "-l", "-b", "-q",
        f"{CONFIG['settings']['framework']}/framework/bbfitting_and_qa/plotSkimTreeQA2D_modified.C(\"{args.config}\")"
    ], check=True)

    LOG.framework("--- plotSkimTreeQA2D_modified.C finished ---")
    LOG.framework("--- Starting fitNormGraphdEdxvsBGpid_modified.C ---")

    subprocess.run([
        "singularity", "exec",
        "/lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif",
        "root", "-l", "-b", "-q",
        f"{CONFIG['settings']['framework']}/framework/bbfitting_and_qa/fitNormGraphdEdxvsBGpid_modified.C(\"{args.config}\")"
    ], check=True)

    LOG.framework("--- fitNormGraphdEdxvsBGpid_modified.C finished ---")
    LOG.framework("--- Starting shift_nsigma_modified.py ---")
    # args.config = "/lustre/alice/users/csonnab/TPC/tpcpid-github-official/output/LHC24/pass1/ar/LHC24ar_pass1_Remove_lustre_TPCSignal_HR_True/20251127/configuration.json"
    # config = read_config(path=args.config)

    subprocess.run([
        "singularity", "exec",
        "/lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif",
        "python3",
        f"{CONFIG['settings']['framework']}/framework/bbfitting_and_qa/shift_nsigma_modified.py",
        "--config", args.config
    ], check=True)

    LOG.framework("--- shift_nsigma_modified.py finished ---")
    LOG.framework("--- Starting CreateDataset.py ---")

    subprocess.run([
        "singularity", "exec",
        "/lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif",
        "python3",
        f"{CONFIG['settings']['framework']}/framework/bbfitting_and_qa/CreateDataset.py",
        "--config", args.config
    ], check=True)

    LOG.framework("--- CreateDataset.py finished ---")
    LOG.framework("All steps completed successfully! Continuing with NN training")

    # args.config = "/lustre/alice/users/csonnab/TPC/tpcpid-github-official/output/LHC23/pass5/zzh/LHC23zzh_pass5_First_FullTest_TPCSignal_HR_True/20251128/configuration.json"
    # config = read_config(path=args.config)
    subprocess.run([
        "python3",
        f"{CONFIG['settings']['framework']}/framework/training_neural_networks/create_jobs.py",
        "--config", args.config,
        "--avoid-question", "1"
    ], check=True)

    subprocess.run([
        "python3",
        f"{CONFIG['settings']['framework']}/framework/training_neural_networks/run_jobs.py",
        "--config", args.config
    ], check=True)



except KeyboardInterrupt:
    LOG.fatal("Interrupted by user. Stopping further execution.")