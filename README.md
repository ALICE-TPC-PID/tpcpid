# tpcpid
Framework for the PID calibration of the ALICE TPC in Run 3
@Jonathan Witte, 01.12.2025
The structure of this directory is as follows:

BasicQA: Scripts that are used to perform the basic QA and put it on the website. Last detailed revision of the scripts: Senjie Zhu.
BBFittingAndQA: Contains all scripts that are used to do a first QA of the input tree, fit the Bethe Bloch curve to the data, create an updated tree with recalculated Nsigma values from the new BB, and create a training dataset for the neural network training. Last detailed revision of the scripts: Jonathan Witte.
    The scripts in this folder are submitted by the "Running" functionality.
NeuralNetworkClass: Contains information and scripts needed for the neural network training and the setup of the network. Setup by: Christian Sonnabend.
    The scripts in this folder are called and handled by the Training-Neural-Networks functionality.
Training-Neural-Networks: Contains the scripts to create the jobs and run the neural network training. Last detailed revision of the scripts: Jonathan Witte/Christian Sonnabend.
    The scripts are colled by the "Running" functionality.
utils: Contains utility scripts that are used in the other folders and contains containers with software for the other scripts.
Running: Contains one default configuration for reference and one configuration.json in use. The script run.sh will use this config to setup a workflow with setup.py, copy all scripts, and submit the jobs. This is used for production. 

To create BB fits and train a neural network, the procedure is the following:
1. Setup the configuration in Running/configuration.json
    Add year, period, pass, input file path and two optional tags for the dataset. 
    Decide to use the dEdx values from the "TPCSignal" tree or the "dEdxNorm" tree.
    Decide if the Hadronic Rate should be used in the training by typing "true" or "false". IMPORTANT: If false is chosen, the Hadronic Rate branch also needs to be removed from the "Labels_x".
    Another useful option is CONFIG["createTrainingDatasetOptions"]["samplesize"] to choose the number of tracks in the training dataset. For debugging, the number of training epochs in the NN training can be set to a small number (f.e. 5) in CONFIG["trainNeuralNetOptions"]["numberOfEpochs"].
2. Run Running/run.sh with "bash run.sh"

The following will be done automatically:
    1. An output folder will be created and all scripts will be copied there. 
    2. One job will be submitted to main with all steps in BBFittingAndQA.
    3. Four jobs will be submitted with the NN training and the training qa. 

Remarks:
    The first job is quite heavy. The default configuration sets the RAM to 32 GB and uses the partition main. However, for input trees above 10 GB, it might be necessary to increase the RAM. The maximum resources needed should be use of the "high_mem" partition with 256 GB RAM (for trees > 30 GB). For debugging, the partition can be set to debug, with time limit reduced to 30 and RAM still at 32 GB.