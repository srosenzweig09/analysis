"""This script was generated in order to HADD together all of the ROOT files generated from the systematic variations.
"""

import shlex
import subprocess
import sys

root = 'root://cmseos.fnal.gov/'
sys_dir = '/store/user/srosenzw/sixb/sixb_ntuples/Summer2018UL/dHHH_pairs/NMSSM/syst/'
cmd = f'eos {root} ls {sys_dir}'
output = subprocess.check_output(shlex.split(cmd))
systematics = output.decode("utf-8").split('\n')[:-1]
dir_path = root + sys_dir

for syst in systematics:
   print(f"[INFO] Processing systematic: {syst}")
   if syst == 'JER': syst = 'JER/eta'
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
      print(output[0])
      break
      infiles = ' '.join(output)
      cmd = f'hadd {root}{path_up}/{sub_dir}/ntuple.root {infiles}'
      try:
         output = subprocess.check_output(shlex.split(cmd))
         output = output.decode("utf-8").split('\n')[:-1]
      except: pass

      print(f"..down")
      cmd = f'xrdfs {root} ls -u {path_dn}/{sub_dir}/output'
      output = subprocess.check_output(shlex.split(cmd))
      output = output.decode("utf-8").split('\n')[:-1]
      infiles = ' '.join(output)
      cmd = f'hadd {root}{path_dn}/{sub_dir}/ntuple.root {infiles}'
      try:
         output = subprocess.check_output(shlex.split(cmd))
         output = output.decode("utf-8").split('\n')[:-1]
      except: pass
      print()
