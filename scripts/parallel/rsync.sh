#!/bin/sh

#SBATCH --job-name=rsync
#SBATCH --qos=avery
#SBATCH --account=avery
#SBATCH --time=4:00:00
#SBATCH --partition=
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=1-10
#SBATCH --output=slurm_output/slurm-%A_%a.out

# Usage: sh scripts/parallel/rsync.sh filelists/Summer2018UL/central.txt
# Usage: sh scripts/parallel/rsync.sh filelists/Summer2018UL/private.txt
# Description: This script takes one argument - the filelist to submit as a command line argument. The script will read the filenames and modify them appropriately for the rsync call.

filelist="$1"

lpcbase="srosenzw@cmslpc-sl7.fnal.gov:/eos/uscms/."
hpgbase="/cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov"

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

if [ $ID -eq -1 ]; then
  MAX=10
  echo "    sbatch --array=0-$(( ${#fileArray[@]} - 1 )) $0 $@"
  sbatch --array=0-$(( ${#fileArray[@]} - 1 )) $0 $@
  exit
fi

echo rsync -avuRP $lpcbase${fileArray[$ID]#$hpgbase} $hpgbase/./
rsync -avuRP $lpcbase${fileArray[$ID]#$hpgbase} $hpgbase/./



### For running them manually:
# rsync -avuRP srosenzw@cmslpc-sl7.fnal.gov:/eos/uscms/./store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM/*/ntuple.root /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/./
# rsync -avuRP srosenzw@cmslpc-sl7.fnal.gov:/eos/uscms/./store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/*/ntuple.root /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/./
# rsync -avuRP srosenzw@cmslpc-sl7.fnal.gov:/eos/uscms/./store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/syst/*/*/*/ntuple.root /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/./
# rsync -avuRP srosenzw@cmslpc-sl7.fnal.gov:/eos/uscms/./store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/JetHT_Data_UL/ntuple.root /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/./
# rsync -avuRP srosenzw@cmslpc-sl7.fnal.gov:/eos/uscms/./store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/TTJets/*/ntuple.root /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/./

# rsync -avuRP srosenzw@cmslpc-sl7.fnal.gov:/eos/uscms/./store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/NMSSM_XToYHTo6B_MX-1100_MY-900_TuneCP5_13TeV-madgraph-pythia8/ntuple.root /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/./


# rsync -avuRP srosenzw@cmslpc-sl7.fnal.gov:/eos/uscms/./store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM/NMSSM_XYH_YToHH_6b_MX_1100_MY_250_2M/ntuple.root /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/./