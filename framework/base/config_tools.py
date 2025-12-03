import os, json
import importlib.util
from datetime import datetime
import subprocess
import sys

from .logger import *

LOG = logger(min_severity="DEBUG", task_name="config_tools")

def read_config(path="../configuration.json"):
    global CONFIG
    path_config = path
    with open(path_config, "r") as f:
        CONFIG = json.load(f)
    return CONFIG


def write_config(CONFIG, path = "../configuration.json"):
    path_config = path
    with open(path_config, "w") as f:
        json.dump(CONFIG, f, indent=4)


#Reads config and adds the name of the dataset
def add_name_and_path(config):
    # Ensure base_output_folder exists; default to $PWD (fall back to os.getcwd() if not set)
    base_folder = config['settings']['framework']

    ### Enable HadronicRate flag if present in input features
    if "fHadronicRate" in config["createTrainingDatasetOptions"]["labels_x"]:
        config["dataset"]["HadronicRate"] = "True"

    dataset = config.get('dataset', {})
    required_keys = ['year', 'period', 'pass', 'dEdxSelection']
    missing = [key for key in required_keys if key not in dataset]
    if missing:
        raise KeyError(f"Missing dataset keys required for output metadata: {missing}")

    name = f"LHC{dataset['year']}{dataset['period']}_{dataset['pass']}_{dataset['dEdxSelection']}"
    if 'optTag' in dataset and dataset['optTag']:
        name += f"_{dataset['optTag']}"
    if dataset.get("HadronicRate", "False") == "True":
        name += "_HadronicRate"
    output_section = config.setdefault('output', {})
    output_section['name'] = name
    config["output"].setdefault('general', {})
    config["output"]["general"]["base_folder"] = base_folder

    base_output = os.path.join(base_folder, "output")
    date_stamp = datetime.now().strftime("%Y%m%d")
    if not "outputPath" in dataset.keys():
        output_path = os.path.join(
            base_output,
            f"LHC{dataset['year']}",
            f"{dataset['period']}",
            f"{dataset['pass']}",
            name,
            date_stamp,
        )
    else:
        output_path = os.path.join(
            base_output,
            f"{dataset['outputPath']}",
            date_stamp,
        )
    LOG.info(f"Framework path = {base_folder}")
    LOG.info(f"Name of dataset = {name}")
    LOG.info(f"Output path = {output_path}")
    config["output"]["general"]["name"] = name
    config["output"]["general"]["path"] = output_path
    if os.path.exists(output_path):
        LOG.warning(f"Output directory {output_path} already exists. -> Will be overwritten!")
    return config

def can_we_continue():
    response = input("\n--> Do you want to continue? (y/n) ")
    if response != 'y':
        LOG.error("Stopping macro!")
        sys.exit(1)
    print("Continuing...\n")

def create_folders(config):

    outdir = config['output']['general']['path']

    if os.path.exists(outdir):
        os.system(f'rm -rf {outdir}')
    os.makedirs(outdir, exist_ok=True)
    LOG.info(f"Created output folder {outdir}")

    tree_dir = os.path.join(outdir, "trees")
    os.makedirs(tree_dir, exist_ok=True)
    LOG.info(f"Created tree output folder {tree_dir}")
    training_dir = os.path.join(outdir, "training")
    os.makedirs(training_dir, exist_ok=True)
    LOG.info(f"Created training output folder {training_dir}")
    config["output"]["general"]["trees"] = tree_dir
    config["output"]["general"]["training"] = training_dir
    processes = ["skimTreeQA", "fitBBGraph", "createTrainingDataset", "trainNeuralNet"]
    for process in processes:
        if config["process"][process]:
            qa_dir = os.path.join(outdir, "QA", process)
            os.makedirs(qa_dir, exist_ok=True)
            LOG.info(f"Setting up QA plot directory {qa_dir}")
            config["output"].setdefault(process, {})
            config["output"][process]["QApath"] = qa_dir

def copy_config(config):

    os.system(f"cp {config['trainNeuralNetOptions']['configuration']} {os.path.join(config['output']['general']['path'], 'nnconfig.py')}")
    config["trainNeuralNetOptions"]["configuration"] = os.path.join(config['output']['general']['path'], 'nnconfig.py')

    write_config(config, path=os.path.join(config["output"]["general"]["path"], "configuration.json"))
    LOG.info("Copied scripts and config to job directory")
    return os.path.join(config["output"]["general"]["path"], "configuration.json")


def import_from_path(path, module_name=None):
    """
    Import a Python module from a file path or directory containing a Python file.

    Args:
        path (str): Absolute path to:
                    - a .py file, OR
                    - a directory containing a Python file to load
        module_name (str, optional): Desired module name. If None, the filename is used.

    Returns:
        module: Imported Python module object (ready to use).
    """

    # If a directory is given, search for .py file
    if os.path.isdir(path):
        # Prefer configurations.py if present
        cfg = os.path.join(path, "configurations.py")
        if os.path.exists(cfg):
            file_path = cfg
        else:
            # Otherwise load the first .py file we find
            py_files = [f for f in os.listdir(path) if f.endswith(".py")]
            if not py_files:
                raise FileNotFoundError(f"No Python file found in directory: {path}")
            file_path = os.path.join(path, py_files[0])
    else:
        # Direct path to a file
        file_path = path

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Could not find Python file: {file_path}")

    # Derive module name from filename if not provided
    if module_name is None:
        module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Make module name unique so multiple configs can be loaded
    unique_module_name = module_name + "_" + str(abs(hash(file_path)))

    # Import machinery
    spec = importlib.util.spec_from_file_location(unique_module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def determine_scheduler(scheduler=None, verbose=False):

    def test_env(cmd):
        return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell=True)==0

    if not scheduler:
        avail_schedulers = []
        slurm_env = test_env("squeue -u {}".format(os.environ.get("USER")))
        condor_env = test_env("condor_q")
        if slurm_env:
            avail_schedulers.append("slurm")
        if condor_env:
            avail_schedulers.append("htcondor")
        if verbose > 0:
            LOG.info("The following schedulers are available: ", avail_schedulers)
            LOG.info(avail_schedulers[0], "is picked for submission\n")
        return avail_schedulers[0]
    else:
        if verbose > 0:
            LOG.info(scheduler, "is picked for submission\n")
        return scheduler
