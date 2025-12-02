# The ALICE TPC PID framework

Framework for the PID calibration of the ALICE TPC in Run 3 and beyond

This framework perfroms the full calibration, from AO2D data taken from the ALICE asynchronous reconstructions to the calibrated Nsigma corrections. To run this framework:

1. Go into the "run" folder
2. Adjust the configurations in configuration.json and nnconfig.py (see below)
3. Execute the framework: python3 run/run.py --config /path/to/configuration.json

Everythign is steered through the configuration.json. This file will be copied to the output folder and gets updated along the framework evaluation. These are some example settings:

```json
{
    "settings": {
        "framework": "/lustre/alice/users/csonnab/TPC/tpcpid-github-official",
        "git": {
            "checkout": 0,
            "create_diff": 1,
            "remote": "https://github.com/ALICE-TPC-PID/tpcpid.git",
            "branch": "main",
            "commit": "latest",
            "tag": "N/A"
        }
    },
    "createTrainingDatasetOptions": {
        "labels_x": [
            "fTPCInnerParam",
            "fTgl",
            "fSigned1Pt",
            "fMass",
            "fNormMultTPC",
            "fNormNClustersTPC",
            "fFt0Occ"
        ],
        "labels_y": [
            "fTPCSignal",
            "fInvDeDxExpTPC"
        ],
        "samplesize": "10000000",
        "sigmarange": "3"
    },
    "dataset": {
        "dEdxSelection": "TPCSignal",
        "pass": "apass5",
        "period": "zzf",
        "year": "23",
        "optTag": "",
        "input_skimmedtree_path": "/lustre/alice/users/marin/R3A/TPCTreeswithNN/PbPbapass5_HR_251006/TPCTrees_HR_LHC23zzf_251006/AO2D_merge_LHC23zzf.root"
    },
    "general": {
        "V0treename": "O2tpcskimv0wde",
        "tpctoftreename": "O2tpctofskimwde"
    },
    "process": {
        "severity": "INFO",
        "createTrainingDataset": true,
        "fitBBGraph": true,
        "shiftNsigma": true,
        "skimTreeQA": true,
        "trainNeuralNet": true
    },
    "trainNeuralNetOptions": {
        "execution_mode": "FULL",
        "configuration": "/lustre/alice/users/csonnab/TPC/tpcpid-github-official/run/configs/nnconfig.py",
        "training_file": "train_single_sigma.py",
        "numberOfEpochs": "200",
        "num_networks": 1,
        "qa_file": "training_qa.py",
        "enable_qa": "True",
        "save_as_pt": "True",
        "save_as_onnx": "True",
        "save_loss_in_files": "True",
        "slurm": {
            "rocm_container": "/lustre/alice/users/csonnab/TPC/TPC_PRODUCTION/Containers/rocm_torch_env.sif",
            "cuda_container": "/lustre/alice/users/csonnab/TPC/TPC_PRODUCTION/Containers/cuda_torch_env.sif",
            "job-name": "TPCPID_TRAINING",
            "partition": "main",
            "time": 480,
            "device": "MI100_GPU",
            "cpus-per-task": 10,
            "mem": "160G",
            "mail-type": "END,FAIL,INVALID_DEPEND",
            "mail-user": "",
            "ngpus": 8
        },
        "htcondor": {
            "universe": "vanilla",
            "+JobFlavour": "workday",
            "request_cpus": "10",
            "request_memory": "10GB",
            "request_disk": "10GB",
            "notify_user": "christian.sonnabend@cern.ch",
            "notification": "Complete,Error"
        }
    }
}
```

Most important settings

- Everything in ```settings``` specifies framework related settings.
    - ```settings/framework```: Specifies the global, absolute path to the framework
    - ```settings/git```: These specify the settings for later possibility to reproduce the state of the repository at the time that it was run
        - ```settings/git/checkout```: When this flag is true, the ```remote```, ```branch```, ```tag``` and ```commit``` fields will be used to reset the repository to the specified version. This should be the produciton default. This can even be a branch of the private github of a collaborator which makes debugging significantly easier.
        - ```settings/git/create_diff```: This will create a diff file in the output folder and also write information into the git_info.txt file to which tag this diff file was made. Like this the current state of work is stored in relation to an upstream tag and can be restored if needed.
- ```createTrainingDatasetOptions``` describe the input and output variables and the number of samples that are being produced for the training data creation from an input file (specified in ```dataset/input_skimmedtree_path```). Normalizations of the input variables is performed in framework/bbfitting_and_qa/CreateDataset.py
- ```dataset```: Description of the data which is to be processed. This will auto-generate the output folder name. You can overwrite the output folder name using ```dataset/outputPath```, by specifying the full path there (not recommended, only for debug purposes)
- ```process```: Describes the processes which are to be executed. These are used in ```run/src/run_framework.py```
- ```trainNeuralNetOptions```: Specifies all the options which are used to train the neural networks
    - Set the path to the nnconfig.py (```trainNeuralNetOptions/configuration```) correctly. You will not need to modify it, the default settings should be fine for running.


~ Christian Sonnabend, Jonathan Witte (Dec. 2025)