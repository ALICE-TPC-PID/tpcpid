import json
import os
from datetime import datetime


def read_config(path="../configuration.json"):  
    global CONFIG
    path_config = path
    with open(path_config, "r") as f:
        CONFIG = json.load(f)
    return CONFIG


def write_config(CONFIG, path = "../configuration.json"):
    path_config = "../configuration.json"
    with open(path_config, "w") as f:
        json.dump(CONFIG, f, indent=4)


#Reads config and adds the name of the dataset
def add_name_and_path(config):
    # Ensure base_output_folder exists; default to $PWD (fall back to os.getcwd() if not set)
    base_folder = os.path.abspath(os.path.join(os.environ.get("PWD", os.getcwd()), ".."))
    dataset = config.get('dataset', {})
    required_keys = ['year', 'period', 'pass', 'optTag1', 'optTag2', 'dEdxSelection', 'HadronicRate']
    missing = [key for key in required_keys if key not in dataset]
    if missing:
        raise KeyError(f"Missing dataset keys required for output metadata: {missing}")

    name = f"LHC{dataset['year']}{dataset['period']}_pass{dataset['pass']}_{dataset['optTag1']}_{dataset['optTag2']}_{dataset['dEdxSelection']}_HR_{dataset['HadronicRate']}"
    output_section = config.setdefault('output', {})
    output_section['name'] = name
    config["output"].setdefault('general', {})
    config["output"]["general"]["base_folder"] = base_folder

    base_output = os.path.join(base_folder, "output")
    date_stamp = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(
        base_output,
        f"LHC{dataset['year']}",
        f"pass{dataset['pass']}",
        f"{dataset['period']}",
        name,
        date_stamp,
    )
    os.makedirs(output_path, exist_ok=True)
    print(f"Name of dataset = {name}")
    print(f"Base path = {base_folder}")
    print(f"Output path = {base_output}")
    config["output"]["general"]["name"] = name
    config["output"]["general"]["path"] = output_path
    return config

def create_folders(config):
    outdir = config['output']['general']['path']
    tree_dir = os.path.join(outdir, "trees")
    os.makedirs(tree_dir, exist_ok=True)
    print(f"[CreateFolders]: Created tree output folder {tree_dir}")
    training_dir = os.path.join(outdir, "training")
    os.makedirs(training_dir, exist_ok=True)
    print(f"[CreateFolders]: Created training output folder {training_dir}")
    job_dir = os.path.join(outdir, "job_scripts")
    os.makedirs(job_dir, exist_ok=True)
    print(f"[CreateFolders]: Created Job folder with scripts {job_dir}")
    config["output"]["general"]["trees"] = tree_dir
    config["output"]["general"]["training"] = training_dir
    config["output"]["general"]["job"] = job_dir
    processes = ["skimTreeQA", "fitBBGraph", "createTrainingDataset", "trainNeuralNet"]
    for process in processes:
        if config["process"][process]:
            qa_dir = os.path.join(outdir, "QA", process)
            os.makedirs(qa_dir, exist_ok=True)
            print(f"[CreateFolders]: Setting up QA plot directory {qa_dir}")
            config["output"].setdefault(process, {})
            config["output"][process]["QApath"] = qa_dir

def copy_scripts(config):
    bbfitting_path = os.path.join(config["output"]["general"]["base_folder"], "BBFittingAndQA")
    NNtraining_path = os.path.join(config["output"]["general"]["base_folder"], "Training-Neural-Networks")
    os.system('cp {0} {1}'.format(os.path.join(bbfitting_path,"*.C"), config["output"]["general"]["job"]))
    os.system('cp {0} {1}'.format(os.path.join(bbfitting_path,"*.py"), config["output"]["general"]["job"]))
    os.system('cp {0} {1}'.format(os.path.join(NNtraining_path,"*.py"), config["output"]["general"]["job"]))
    os.system('cp {0} {1}'.format(os.path.join(config["output"]["general"]["base_folder"],"utils","config_tools.py"), config["output"]["general"]["job"]))
    os.system('cp {0} {1}'.format("configuration.json", config["output"]["general"]["path"]))
    
    print("Copied scripts and config to job directory")