#!/bin/bash

cd running
python3 run.py --config configuration_default.json
cd ..
exit 0
singularity exec /lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif \
root -l -b -q 'framework/bbfitting_and_qa/plotSkimTreeQA2D_modified.C()'
singularity exec /lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif \
root -l -b -q 'framework/bbfitting_and_qa/fitNormGraphdEdxvsBGpid_modified.C'
singularity exec /lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif python3 framework/bbfitting_and_qa/shift_nsigma_modified.py
source /lustre/alice/users/jwitte/tpcpid/pythonenv/EnvironmentforNNDataset/bin/activate
/lustre/alice/users/jwitte/pythonenv/EnvironmentforNNDataset/bin/python3 framework/bbfitting_and_qa/CreateDataset.py
cd framework/training_neural_networks/
singularity exec /lustre/alice/users/jwitte/singularity/python_hipe4ml_root.sif python3 create_jobs.py