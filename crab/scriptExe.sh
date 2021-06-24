#!/bin/bash
BASE=$PWD

echo "================= CMSRUN starting jobNum=$1 ====================" | tee -a job.log
# export SCRAM_ARCH=slc6_amd64_gcc700
export SCRAM_ARCH=slc7_amd64_gcc700

echo "================= CMSRUN starting GEN-SIM step ====================" | tee -a job.log
# cmsRun -j genSim_step.log genSim_step.py jobNum=$1 nEvents=500 
# python train_nn.py --tag testing 
touch test_file.py