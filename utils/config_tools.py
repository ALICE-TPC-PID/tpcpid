import json
import os
from datetime import datetime


def read_config():  
    global CONFIG
    path_config = "../Running/configuration.json"
    with open(path_config, "r") as f:
        CONFIG = json.load(f)
    return CONFIG


def write_config(CONFIG):
    path_config = "../Running/configuration.json"
    with open(path_config, "w") as f:
        json.dump(CONFIG, f, indent=4)


def ensure_output_metadata(config):
    """Create dataset name and output folder, storing both under config['output']."""
    dataset = config.get('dataset', {})
    required_keys = ['year', 'period', 'pass', 'optTag1', 'optTag2', 'dEdxSelection', 'HadronicRate']
    missing = [key for key in required_keys if key not in dataset]
    if missing:
        raise KeyError(f"Missing dataset keys required for output metadata: {missing}")

    name = f"LHC{dataset['year']}{dataset['period']}_pass{dataset['pass']}_{dataset['optTag1']}_{dataset['optTag2']}_{dataset['dEdxSelection']}_HR_{dataset['HadronicRate']}"
    output_section = config.setdefault('output', {})
    output_section['name'] = name

    base_output = config["general"]["base_output_folder"]
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

    return name, output_path


#Reads config and adds the name of the dataset
def add_name_and_path(config):
    name, output_path = ensure_output_metadata(config)
    print(f"Name of dataset = {name}")
    print(f"Output path = {output_path}")
    config["output"].setdefault('general', {})
    config["output"]["general"]["name"] = name
    config["output"]["general"]["path"] = output_path
    return config

def create_folders(config):
    outdir = config['output']['general']['path']
    tree_dir = os.path.join(outdir, "trees")
    os.makedirs(tree_dir, exist_ok=True)
    print(f"[CreateFolders]: Created tree output folder {tree_dir}")
    config["output"]["general"]["trees"] = tree_dir
    processes = ["skimTreeQA", "fitBBGraph", "createTrainingDataset", "trainNeuralNet"]
    for process in processes:
        if config["process"][process]:
            qa_dir = os.path.join(outdir, "QA", process)
            os.makedirs(qa_dir, exist_ok=True)
            print(f"[CreateFolders]: Setting up QA plot directory {qa_dir}")
            config["output"].setdefault(process, {})
            config["output"][process]["QApath"] = qa_dir

