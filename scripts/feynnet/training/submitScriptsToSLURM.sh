#!/bin/sh

#SBATCH --job-name=train-count
#SBATCH --qos=avery
#SBATCH --account=avery
#SBATCH --time=4:00:00
#SBATCH --partition=hpg-dev
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=1-10
#SBATCH --output=slurm_output/slurm-%A_%a.out
#SBATCH --requeue

if [ -z "$ID" ]; then

  # get the job id
  ID=$SLURM_ARRAY_TASK_ID

  # if the job id is empty, exit
  if [ -z "$ID" ]; then
    ID=-1
  fi

fi

# filelist=filelists/Summer2018UL/central.txt
filelist=filelists/Summer2018UL/private.txt

NBATCHES=10

if [ $ID -eq -1 ]; then
  cmd="sbatch -W --array=0-$(( $NBATCHES - 1 )) $0 $@"
  echo "    $cmd"
  $cmd
  exit
fi

python scripts/feynnet/training/calculate_nevents.py $filelist --nbatches $NBATCHES --batch $ID
