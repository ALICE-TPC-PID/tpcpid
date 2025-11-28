#!/bin/bash
# Usage: enter an O2Physics environment, then  ./mergeSkimTrees.sh periodName passName

if [ -z "${O2PHYSICS_ROOT}" ]; then
   echo "No O2Physics environment detected. Quitting..."
   exit 1
fi

periodName=$1
periodYear="20"$(expr substr ${periodName} 4 2)
passName=$2

if [[ ${periodName} != LHC* ]]; then
   echo "Period name ${periodName} not recognised (should be e.g. 'LHC23xyz'). Quitting..."
   exit 1;
fi

read -r -p "Downloading and merging trees for period name: ${periodName}. Continue? [y/N] " response
if [[ "$response" =~ ^[yY].* ]]; then
   
   alien_cp /alice/data/$periodYear/${periodName} -name ends_[0-9]*/${passName}/[0-9]*/OfflineTriggerSelectionCalibration2023/AOD/[0-9]*/tpc_skims.root file:.
   find ./${periodName} -name tpc_skims.root > treeNames${periodName}_${passName}.txt
   o2-aod-merger --input treeNames${periodName}_${passName}.txt --output tpc_skims_merged_${periodName}_${passName}.root --max-size 100000000000
   
   alien_cp /alice/data/$periodYear/${periodName} -name ends_[0-9]*/${passName}/[0-9]*/OfflineTriggerSelectionCalibration2023/Stage.5/[0-9]*/AnalysisResults.root file:.
   find ./${periodName} -name AnalysisResults.root | xargs hadd AnalysisResults_merge_${periodName}_${passName}.root 
   
else
   echo "Cancelling"
   exit 0
fi

exit 0
