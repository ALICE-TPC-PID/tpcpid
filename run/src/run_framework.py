import os, sys, json, subprocess, shlex
from argparse import ArgumentParser

parser = ArgumentParser(description="Setup the TPC PID analysis")
parser.add_argument("-c", "--config", type=str, default="configuration.json", help="Path to configuration file")
parser.add_argument("-ci", "--ci-run", type=int, default=0, help="Run in CI mode")
args = parser.parse_args()

args.config = os.path.abspath(args.config)

with open(args.config, 'r') as config_file:
    CONFIG = json.load(config_file)

sys.path.append(CONFIG['settings']['framework'] + "/framework")
from base import *

epn_switch = "EPN" in CONFIG["trainNeuralNetOptions"]["slurm"]["device"]


def run_cmd(command, workdir=None, barerun=False):
    """
    CI: run directly
    EPN: module load O2PDPSuite once per subprocess shell, then run directly
    non-EPN: run inside Apptainer
    """
    if args.ci_run or barerun:
        subprocess.run(command, check=True, cwd=workdir)
        return

    if epn_switch:
        quoted_cmd = " ".join(shlex.quote(str(x)) for x in command)
        shell_cmd = f"module load O2PDPSuite && {quoted_cmd}"
        subprocess.run(["bash", "-lc", shell_cmd], check=True, cwd=workdir)
        return

    subprocess.run([
        "apptainer", "exec",
        CONFIG["settings"]["base_container"],
        *command
    ], check=True, cwd=workdir)


try:
    LOG = logger("Framework")
    LOG.framework("Setup completed successfully. Ready to launch!")

    if not args.ci_run:
        full_git_config(
            save_to_file=os.path.join(CONFIG["output"]["general"]["path"], "git_info.txt"),
            verbose=False,
            path=CONFIG['settings']['framework']
        )

        if CONFIG["settings"]["git"].get("create_diff", False):
            diff, repo_url, tag = diff_to_latest_upstream_tag(
                path=CONFIG["settings"]["framework"],
                diff_file=os.path.join(CONFIG["output"]["general"]["path"], "git_diff.patch"),
                info_file=os.path.join(CONFIG["output"]["general"]["path"], "git_info.txt")
            )

    if CONFIG["process"]["skimTreeQA"]:
        LOG.framework("--- Starting plotSkimTreeQA2D_modified.C ---")

        run_cmd([
            "root", "-l", "-b", "-q",
            f'{CONFIG["settings"]["framework"]}/framework/bbfitting_and_qa/plotSkimTreeQA2D_modified.C("{args.config}")'
        ])

        LOG.framework("--- plotSkimTreeQA2D_modified.C finished ---")

    if CONFIG["process"]["electronCleaning"] and (not args.ci_run):
        LOG.framework("--- Starting tmva_application.py ---")

        electron_cleaning_dir = os.path.join(
            CONFIG["output"]["general"]["path"],
            "electronCleaning"
        )

        run_cmd([
            "root", "-l", "-b", "-q",
            f'{CONFIG["settings"]["framework"]}/framework/electron_cleaning/Train.cpp("{CONFIG["dataset"]["input_skimmedtree_path"]}", "{electron_cleaning_dir}/TMVAC.root", "bdt")'
        ], workdir=electron_cleaning_dir)

        run_cmd([
            "python3",
            f'{CONFIG["settings"]["framework"]}/framework/electron_cleaning/tmva_application.py',
            "--config", args.config
        ])

        LOG.framework("--- tmva_application.py finished ---")

    if CONFIG["process"]["fitBBGraph"]:
        LOG.framework("--- Starting fitNormGraphdEdxvsBGpid_modified.C ---")

        run_cmd([
            "root", "-l", "-b", "-q",
            f'{CONFIG["settings"]["framework"]}/framework/bbfitting_and_qa/fitNormGraphdEdxvsBGpid_modified.C("{args.config}")'
        ])

        LOG.framework("--- fitNormGraphdEdxvsBGpid_modified.C finished ---")

    if CONFIG["process"]["shiftNsigma"]:
        LOG.framework("--- Starting shift_nsigma_modified.py ---")

        run_cmd([
            "python3",
            f'{CONFIG["settings"]["framework"]}/framework/bbfitting_and_qa/shift_nsigma_modified.py',
            "--config", args.config
        ])

        LOG.framework("--- shift_nsigma_modified.py finished ---")

    if CONFIG["process"]["createTrainingDataset"]:
        LOG.framework("--- Starting CreateDataset.py ---")

        run_cmd([
            "python3",
            f'{CONFIG["settings"]["framework"]}/framework/bbfitting_and_qa/CreateDataset.py',
            "--config", args.config
        ], barerun=True)

        LOG.framework("--- CreateDataset.py finished ---")

    LOG.framework("All steps completed successfully! Continuing with NN training")

    if CONFIG["process"]["trainNeuralNet"]:
        subprocess.run([
            "python3",
            f'{CONFIG["settings"]["framework"]}/framework/training_neural_networks/create_jobs.py',
            "--config", args.config,
            "--avoid-question", "1"
        ], check=True)

        subprocess.run([
            "python3",
            f'{CONFIG["settings"]["framework"]}/framework/training_neural_networks/run_jobs.py',
            "--config", args.config,
            "--ci-run", str(args.ci_run)
        ], check=True)

except KeyboardInterrupt:
    LOG.fatal("Interrupted by user. Stopping further execution.")