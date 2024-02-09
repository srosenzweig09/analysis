#!/usr/bin/env python

"""
python scripts/hadd_eos.py --tag <tag> --sample <sample>

<sample> :
         data
         NMSSM
         mc_bkg
<tag> :
         bias
         btag
         dnn
         gnn
         nocuts
         presel
"""

import re, shlex, subprocess, sys
from argparse import ArgumentParser
import sys

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--sample', dest='sample', required=True)
parser.add_argument('--tag', dest='tag', required=True)

args = parser.parse_args()

root = 'root://cmseos.fnal.gov'

error_text = "File exists\n\nError in <TFileMerger::OutputFile>: cannot open the MERGER output file \nhadd error opening target file"

def hadd_NMSSM():
   outdir=f'/store/user/srosenzw/sixb/ntuples/Summer2018UL/{args.tag}/'
   cmd = f'xrdfs {root} ls -u {outdir}'
   output = subprocess.check_output(shlex.split(cmd))
   output = output.decode("utf-8").split('\n')

   for line in output:
      print(line)
      if 'NMSSM_XYH_YToHH' not in line: continue
      # if '10M' in line: continue
      # if '2M' in line: continue
      # if '100k' in line: continue
      sample = line.split('/')[-1]
      # print(sample)
      match = re.search('/store/', line)
      start = match.start()
      end = match.end()
      cmd = f'xrdfs {root} ls -u {line[start:]}/output'
      output = subprocess.check_output(shlex.split(cmd))
      output = output.decode("utf-8").split('\n')
      infiles = ' '.join(output)
      cmd = f'hadd -f {root}/{line[start:]}/ntuple.root {infiles}'
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

def hadd_qcd_ttbar():
   outdir=f'/store/user/srosenzw/sixb/ntuples/Summer2018UL/{args.tag}/QCD'

   cmd = f'xrdfs {root} ls -u {outdir}'
   subdirs = subprocess.check_output(shlex.split(cmd))
   subdirs = subdirs.decode("utf-8").split('\n')
   for subdir in subdirs: 
      print(subdir.split('//')[2])
      cmd = f'xrdfs {root} ls -u /{subdir}'
      output = subprocess.check_output(shlex.split(cmd))
      output = output.decode("utf-8").split('\n')
      print(output)
   # subdirs = ' '.join(subdirs)
   # print(subdirs)
   # cmd = f'hadd {root}/{outdir}/ntuple.root {infiles}'
   # subprocess.run(shlex.split(cmd))

if args.sample == 'data': hadd_data()
if args.sample =='mc_bkg': hadd_qcd_ttbar()
if args.sample == 'NMSSM': hadd_NMSSM()
# if args.dnn: hadd_NMSSM('dnn')