#!/bin/bash



##########################################################################################################
# To run this script:                                                                                    #
# - enter an O2Physics environment                                                                       #
# - click on the "Copy all output directories" button on the "Submitted jobs" tab of the train page,     #   
#   and paste the contents into a file named "outputDirectories.txt" in the same folder                  #
# - run with ./copyTreesFromHyperloop.sh                                                                 #
# - Output will be stored as "SkimmedTree_merged.root", along with the intermediate files in run-number  #
#   folders.                                                                                             #
##########################################################################################################

if [[ $1 != --skip-download ]]; then

   for path in `sed 's/,/\n/g' outputDirectories.txt`; do
      # Get period name + run number from attached "analysis.xml"
      periodNameAndRunNumber=`stdbuf -oL alien.py grep -o "LHC[0-9]*[a-z]*/[0-9]*" $path/analysis.xml | head -n 1`
      echo "creating output directory for period/run $periodNameAndRunNumber"
      mkdir -p $periodNameAndRunNumber
      
      if [[ $(alien_find -r $path AOD/[0-9]*/AO2D.root) ]];  then   # Use "AOD" partial merges if available
         echo "found merged files in AOD subfolder for $path, starting copy . . ."
         alien_cp -name ends_AOD/[0-9]*/[a-zA-Z0-9]*.root $path file:$periodNameAndRunNumber
      else 
         echo "No AOD subfolder found, falling back to parent directory"
         alien_cp -name ends_[0-9]*/[a-zA-Z0-9]*.root $path file:$periodNameAndRunNumber
      fi 
   
   done
else
   echo "Skipping download due to --skip-download switch . . ."
fi


for period in `ls -d LHC[0-9]*[a-z]*`; do
   # Find AO2D files and create merge list
   find $period -name AO2D.root > mergelist_$period.txt
   echo "Starting merge for AO2Ds in $period"
   if [ -f "AO2D_merge_$period.root" ]; then
      echo "Old output found, cleaning up first"
      rm AO2D_merge_$period.root
   fi
   o2-aod-merger --input mergelist_$period.txt --output AO2D_merge_$period.root --max-size 10000000000

   echo "Merging AnalysisResults files for $period"
   if [ -f "AnalysisResults_merge_$period.root" ]; then
      echo "Old output found, cleaning up first"
      rm AnalysisResults_merge_$period.root
   fi
   find $period -name AnalysisResults.root | xargs hadd AnalysisResults_merge_$period.root
done



