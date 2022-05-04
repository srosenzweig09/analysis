"""This script was generated in order to HADD together all of the ROOT files generated from the systematic variations.
"""

from argparse import ArgumentParser
import shlex
import subprocess
import sys

parser = ArgumentParser(description='Command line parser of model options and tags')
parser.add_argument('--bias', dest='bias', help='', action='store_true', default=False)
parser.add_argument('--btag', dest='btag', help='', action='store_true', default=False)
args = parser.parse_args()

if args.bias: tag = 'dHHH_pairs'
elif args.btag: tag = 'dHHH_pairs_maxbtag'
else:
   raise "Please provide --btag or --bias!"

root = 'root://cmseos.fnal.gov/'
sys_dir = f'/store/user/srosenzw/sixb/sixb_ntuples/Summer2018UL/{tag}/NMSSM/syst/'
cmd = f'eos {root} ls {sys_dir}'
output = subprocess.check_output(shlex.split(cmd))
systematics = output.decode("utf-8").split('\n')[:-1]
dir_path = root + sys_dir

for syst in systematics:
   print(f"[INFO] Processing systematic: {syst}")
   if syst != 'HF': continue
   if syst == 'JER': syst = 'JER/pt'
   path = sys_dir + syst + '/'
   path_up = path + 'up'
   path_dn = path + 'down'
   # print(path_up + '\n' + path_dn)
   cmd = f'eos {root} ls {path_up}'
   output = subprocess.check_output(shlex.split(cmd))
   sub_dirs = output.decode("utf-8").split('\n')[:-2]

   for sub_dir in sub_dirs:
      print(sub_dir)
      print(f"..up")
      cmd = f'xrdfs {root} ls -u {path_up}/{sub_dir}/output'
      output = subprocess.check_output(shlex.split(cmd))
      output = output.decode("utf-8").split('\n')[:-1]
      infiles = ' '.join(output)
      cmd = f'hadd -f {root}{path_up}/{sub_dir}/ntuple.root {infiles}'
      try:
         output = subprocess.check_output(shlex.split(cmd))
         output = output.decode("utf-8").split('\n')[:-1]
      except: pass

      print(f"..down")
      cmd = f'xrdfs {root} ls -u {path_dn}/{sub_dir}/output'
      output = subprocess.check_output(shlex.split(cmd))
      output = output.decode("utf-8").split('\n')[:-1]
      infiles = ' '.join(output)
      cmd = f'hadd -f {root}{path_dn}/{sub_dir}/ntuple.root {infiles}'
      try:
         output = subprocess.check_output(shlex.split(cmd))
         output = output.decode("utf-8").split('\n')[:-1]
      except: pass
      print()
