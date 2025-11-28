import os, sys, pathlib, subprocess
from argparse import ArgumentParser
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/framework")

from framework import base

LOG = base.logger.logger("Framework")

try:
    parser = ArgumentParser(description="Setup the TPC PID analysis")
    parser.add_argument("-c", "--config", type=str, default="configuration.json",
                        help="Path to configuration file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        LOG.fatal(f"Configuration file {args.config} not found. Aborting")
        sys.exit(1)

    base.full_git_config(save_to_file=False, verbose=True)

    config = base.read_config(path=args.config)
    config = base.add_name_and_path(config)
    base.create_folders(config)
    config_path = base.copy_config(config)
    LOG.info("New config path is " + config_path)

    base.full_git_config(
        save_to_file=os.path.join(config["output"]["general"]["path"], "git_info.txt"),
        verbose=False
    )

    LOG.info("Setup completed successfully. Ready to launch!")
    LOG.info("--- Starting plotSkimTreeQA2D_modified.C ---")

    subprocess.run([
        "singularity", "exec",
        "/lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif",
        "root", "-l", "-b", "-q",
        f"{config['output']['general']['base_folder']}/framework/bbfitting_and_qa/plotSkimTreeQA2D_modified.C(\"{config_path}\")"
    ], check=True)

    LOG.info("--- plotSkimTreeQA2D_modified.C finished ---")
    LOG.info("--- Starting fitNormGraphdEdxvsBGpid_modified.C ---")

    subprocess.run([
        "singularity", "exec",
        "/lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif",
        "root", "-l", "-b", "-q",
        f"{config['output']['general']['base_folder']}/framework/bbfitting_and_qa/fitNormGraphdEdxvsBGpid_modified.C(\"{config_path}\")"
    ], check=True)

    LOG.info("--- fitNormGraphdEdxvsBGpid_modified.C finished ---")
    LOG.info("--- Starting shift_nsigma_modified.py ---")
    # config_path = "/lustre/alice/users/csonnab/TPC/tpcpid-github-official/output/LHC24/pass1/ar/LHC24ar_pass1_Remove_lustre_TPCSignal_HR_True/20251127/configuration.json"
    # config = base.read_config(path=config_path)

    subprocess.run([
        "singularity", "exec",
        "/lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif",
        "python3",
        f"{config['output']['general']['base_folder']}/framework/bbfitting_and_qa/shift_nsigma_modified.py",
        "--config", config_path
    ], check=True)

    LOG.info("--- shift_nsigma_modified.py finished ---")
    LOG.info("--- Starting CreateDataset.py ---")

    subprocess.run([
        "singularity", "exec",
        "/lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif",
        "python3",
        f"{config['output']['general']['base_folder']}/framework/bbfitting_and_qa/CreateDataset.py",
        "--config", config_path
    ], check=True)

    LOG.info("--- CreateDataset.py finished ---")
    LOG.info("All steps completed successfully! Continuing with NN training")

    config_path = "/lustre/alice/users/csonnab/TPC/tpcpid-github-official/output/LHC23/pass5/zzh/LHC23zzh_pass5_First_FullTest_TPCSignal_HR_True/20251128/configuration.json"
    config = base.read_config(path=config_path)
    subprocess.run([
        "python3",
        f"{config['output']['general']['base_folder']}/framework/training_neural_networks/create_jobs.py",
        "--config", config_path,
        "--avoid-question", "1"
    ], check=True)

    subprocess.run([
        "python3",
        f"{config['output']['general']['base_folder']}/framework/training_neural_networks/run_jobs.py",
        "--config", config_path
    ], check=True)



except KeyboardInterrupt:
    LOG.fatal("Interrupted by user. Stopping further execution.")