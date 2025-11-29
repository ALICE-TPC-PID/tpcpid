In this folder the full functionality of running the BB fitting, NN training, with all QA should live. It will have a subdirectory for the JOBs created to run. 
It will contain the scripts, that are needed for running.
It will contain the blanko configuration file.
The configuration file will steer which of the steps are performed. It will also contain all parameters, cuts, normalisations etc, which are used during the fitting and training
The QA as well as all output will be stored not in here.

I might need to setup a support script that creates the name of the dataset as well as the output folder and structures. It should also create a auxiliary.json, in which the name of the dataset as well as f.e. the BB parameters are stored. This should already live in the output folder.