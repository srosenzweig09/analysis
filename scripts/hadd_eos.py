#!/usr/bin/env python

import re, shlex, subprocess, sys
from argparse import ArgumentParser

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--data', dest='data', action='store_true', default=False)
parser.add_argument('--bias', dest='bias', action='store_true', default=False)
parser.add_argument('--btag', dest='btag', action='store_true', default=False)
parser.add_argument('--nocuts', dest='nocuts', action='store_true', default=False)
parser.add_argument('--presel', dest='presel', action='store_true', default=False)

args = parser.parse_args()

root = 'root://cmseos.fnal.gov'

error_text = "File exists\n\nError in <TFileMerger::OutputFile>: cannot open the MERGER output file \nhadd error opening target file"

def hadd_NMSSM(tag):
   outdir=f'/store/user/srosenzw/sixb/ntuples/Summer2018UL/{tag}/'
   cmd = f'xrdfs {root} ls -u {outdir}'
   output = subprocess.check_output(shlex.split(cmd))
   output = output.decode("utf-8").split('\n')

   for line in output:
      print(line)
      if 'NMSSM_XYH_YToHH' not in line: continue
      sample = line.split('/')[-1]
      # print(sample)
      match = re.search('/store/', line)
      start = match.start()
      end = match.end()
      cmd = f'xrdfs {root} ls -u {line[start:]}/output'
      output = subprocess.check_output(shlex.split(cmd))
      output = output.decode("utf-8").split('\n')
      infiles = ' '.join(output)
      cmd = f'hadd {root}/{line[start:]}/ntuple.root {infiles}'
      proc = subprocess.Popen(shlex.split(cmd),stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      (stdout, stderr) = proc.communicate()

      if error_text in stderr.decode("UTF-8"):
         print("[WARNING] File already exists!")
      elif 'usage' in stderr.decode("UTF-8"):
         print("[ERROR] Output files may not exist!")
      else:
         print("[INFO] Successfully hadded!")
      print()

def hadd_data():
   outdir=f'/store/user/srosenzw/sixb/sixb_ntuples/Summer2018UL/{args.tag}/JetHT_Run2018_full'
   cmd = f'xrdfs {root} ls -u {outdir}/output'
   output = subprocess.check_output(shlex.split(cmd))
   output = output.decode("utf-8").split('\n')
   infiles = ' '.join(output)
   cmd = f'hadd {root}/{outdir}/ntuple.root {infiles}'
   subprocess.run(shlex.split(cmd))

if args.data: hadd_data()
if args.bias: hadd_NMSSM('bias/NMSSM')
if args.btag: hadd_NMSSM('btag/NMSSM')
if args.nocuts: hadd_NMSSM('NMSSM_nocuts')
if args.presel: hadd_NMSSM('NMSSM_presel')