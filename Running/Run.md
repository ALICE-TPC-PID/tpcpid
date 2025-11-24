#!/bin/bash
source ~/.bashrc
which rootContainer
cd ../BBFittingAndQA
rootContainer -l -b -q 'plotSkimTreeQA2D_modified.C()'
rootContainer -l -b -q 'fitNormGraphdEdxvsBGpid_modified.C'
python3Container shift_nsigma_modified.py
source /lustre/alice/users/jwitte/tpcpid/pythonenv/EnvironmentforNNDataset/bin/activate
/lustre/alice/users/jwitte/pythonenv/EnvironmentforNNDataset/bin/python3 CreateDataset.py