#!/usr/bin/env bash
# Based on misc/test-mlfix/master/run.sh; configuration is prepared by the CI runner.
set -euo pipefail

o2-analysis-pid-tpc-skimscreation -b --configuration json://configuration.json | \
o2-analysis-pid-tof-merge -b --configuration json://configuration.json | \
o2-analysis-multcenttable -b --configuration json://configuration.json | \
o2-analysis-event-selection-service -b --configuration json://configuration.json | \
o2-analysis-propagationservice -b --configuration json://configuration.json | \
o2-analysis-trackselection -b --configuration json://configuration.json | \
o2-analysis-lf-strangenesstofpid -b --configuration json://configuration.json | \
o2-analysis-dq-v0-selector -b --configuration json://configuration.json | \
o2-analysis-pid-tpc-service -b --configuration json://configuration.json | \
o2-analysis-ft0-corrected-table -b --configuration json://configuration.json --aod-file "${O2_AOD_FILE:-/fixtures/AO2D.2dfs.root}" --aod-memory-rate-limit 209715200 --shm-segment-size ${O2_SHM_SIZE:-2000000000} --aod-writer-keep "AOD/TPCTOFSKIMTREE/0,AOD/TPCSKIMV0TREE/0"
