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
    LOG = logger("Framework")
    LOG.framework("Setup completed successfully. Ready to launch!")

    full_git_config(
        save_to_file=os.path.join(CONFIG["output"]["general"]["path"], "git_info.txt"),
        verbose=False,
        path=CONFIG['settings']['framework']
    )

    if CONFIG["settings"]["git"].get("create_diff", False):
        diff, repo_url, tag = diff_to_latest_upstream_tag(path=CONFIG["settings"]["framework"], diff_file=os.path.join(CONFIG["output"]["general"]["path"], "git_diff.patch"), info_file=os.path.join(CONFIG["output"]["general"]["path"], "git_info.txt"))


    if CONFIG["process"]["skimTreeQA"]:
        LOG.framework("--- Starting plotSkimTreeQA2D_modified.C ---")

        subprocess.run([
            "apptainer", "exec",
            "/lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif",
            "root", "-l", "-b", "-q",
            f"{CONFIG['settings']['framework']}/framework/bbfitting_and_qa/plotSkimTreeQA2D_modified.C(\"{args.config}\")"
        ], check=True)

        LOG.framework("--- plotSkimTreeQA2D_modified.C finished ---")


    if CONFIG["process"]["fitBBGraph"]:
        LOG.framework("--- Starting fitNormGraphdEdxvsBGpid_modified.C ---")

        subprocess.run([
            "apptainer", "exec",
            "/lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif",
            "root", "-l", "-b", "-q",
            f"{CONFIG['settings']['framework']}/framework/bbfitting_and_qa/fitNormGraphdEdxvsBGpid_modified.C(\"{args.config}\")"
        ], check=True)

        LOG.framework("--- fitNormGraphdEdxvsBGpid_modified.C finished ---")


    if CONFIG["process"]["shiftNsigma"]:
        LOG.framework("--- Starting shift_nsigma_modified.py ---")

        subprocess.run([
            "apptainer", "exec",
            "/lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif",
            "python3",
            f"{CONFIG['settings']['framework']}/framework/bbfitting_and_qa/shift_nsigma_modified.py",
            "--config", args.config
        ], check=True)

        LOG.framework("--- shift_nsigma_modified.py finished ---")


    if CONFIG["process"]["createTrainingDataset"]:
        LOG.framework("--- Starting CreateDataset.py ---")

        subprocess.run([
            "apptainer", "exec",
            "/lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif",
            "python3",
            f"{CONFIG['settings']['framework']}/framework/bbfitting_and_qa/CreateDataset.py",
            "--config", args.config
        ], check=True)

        LOG.framework("--- CreateDataset.py finished ---")


    LOG.framework("All steps completed successfully! Continuing with NN training")


    if CONFIG["process"]["trainNeuralNet"]:
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