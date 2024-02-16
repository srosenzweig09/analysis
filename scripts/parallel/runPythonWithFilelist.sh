#!/bin/sh

#SBATCH --job-name=sixb
#SBATCH --qos=avery
#SBATCH --account=avery
#SBATCH --time=4:00:00
#SBATCH --partition=
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=
#SBATCH --output=slurm_output/slurm-%A_%a.out

# Usage: parallel_filelist.sh pythonScript.py filelist
# Description: This script takes two arguments, the first of which should be the python script you wish to run, the second being the filelist to submit as a command line argument. 

script="$1"
filelist="$2"

source $HOME/.bashrc
ml conda
mamba activate sixb

hostname
date
pwd
which python

if [ -z "$ID" ]; then

  # get the job id
  ID=$SLURM_ARRAY_TASK_ID

  # if the job id is empty, exit
  if [ -z "$ID" ]; then
    ID=-1
  fi

fi

mapfile -t fileArray < $filelist

echo Number of files in filelist: ${#fileArray[@]}
# echo "Plot directories:"
# for arr in ${fileArray[@]}; do
#   echo "  $PLOT"
# done

if [ $ID -eq -1 ]; then

  MAX=5

  echo "    sbatch --array=0-$(( ${#fileArray[@]} - 1 )) $0 $@"
  sbatch --array=0-$(( ${#fileArray[@]} - 1 )) $0 $@
  exit
fi

echo python $script ${fileArray[$ID]}
python $script ${fileArray[$ID]}

date
