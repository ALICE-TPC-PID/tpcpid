#!/bin/bash
infile=$1

###
# Usage:
# Copy the bullet points "LHCXXX all/all: /alice..." of the "Hyperloop train run finished" e-mail into a text file "input.txt"
# Run this script using ./downloadSkimTreesHyperloop.sh input.txt
# This will parse the paths and put the AnalysisResults.root and AO2D.root (the trees) into subdirectories named for each period.
###


while read -r line; do  
   if [[ "$line" == "LHC"* ]] ; then # update period name
      periodName=${line%% *}   # trim end of line
   fi
   if [[ "$line" == *"/alice/"* ]] ; then #update input path
      if [[ -z ${periodName} ]]; then  # only take LHC periods, not the main merge file
         continue;
      fi
       
      inputPath=${line%%(browse)*} ## get rid of trailing "browse"
      inputPath=${inputPath##*all/all:} ## and leading lines if it's the "job complete" e-mail
      mkdir -p $periodName
      echo $inputPath
      echo $periodName
      alien_cp $inputPath"/" -name ends_.*\.root "file:"$periodName"/"
   fi
done < $infile
