"""This script was generated in order to HADD together all of the ROOT files generated from the systematic variations.
"""

from argparse import ArgumentParser
from colorama import Fore, Style
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
if systematics[-1] == '{systematic}': systematics = systematics[:-1]
print(systematics)
dir_path = root + sys_dir

error_text = "File exists\n\nError in <TFileMerger::OutputFile>: cannot open the MERGER output file \nhadd error opening target file"

def debug_hadd(error):
   error = error.decode("UTF-8")
   if error_text in error:
      print(f"[{Fore.YELLOW}WARNING{Style.RESET_ALL}] File already exists!")
   elif 'usage' in error:
      print(f"[{Fore.YELLOW}WARNING{Style.RESET_ALL}] Output files may not exist!")
      print_error = input("Print error?")
      if print_error.lower() == 'yes': print(error)
      sys.exit()
   elif 'No such file or directory' in stderr.decode("UTF-8"):
      print(f"[{Fore.RED}ERROR{Style.RESET_ALL}] No job found!")
   else:
      print(f"[INFO] {Fore.GREEN}Successfully hadded!{Style.RESET_ALL}")

# flag = True
for syst in systematics:
   print(f"[INFO] Processing systematic: {syst}")
   if 'JER' not in syst: continue
   # if 'EC2_2018' in syst: flag = False
   # if flag: continue
   if syst == 'JER': syst = 'JER/pt'
   path = sys_dir + syst + '/'
   path_up = path + 'up'
   path_dn = path + 'down'
   # print(path_up + '\n' + path_dn)
   cmd = f'eos {root} ls {path_up}'
   output = subprocess.check_output(shlex.split(cmd))
   sub_dirs = output.decode("utf-8").split('\n')[:-2]

   for sub_dir in sub_dirs:
      if sub_dir == 'NMSSM_XYH_YToHH_6b_MX_1200_MY_1000': continue
      print(path)
      print(sub_dir)
      print(f"..up")
      cmd = f'xrdfs {root} ls -u {path_up}/{sub_dir}/output'
      output = subprocess.check_output(shlex.split(cmd))
      output = output.decode("utf-8").split('\n')[:-1]
      infiles = ' '.join(output)
      cmd = f'hadd {root}{path_up}/{sub_dir}/ntuple.root {infiles}'
      proc = subprocess.Popen(shlex.split(cmd),stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      (stdout, stderr) = proc.communicate()

      debug_hadd(stderr)

      print(f"..down")
      cmd = f'xrdfs {root} ls -u {path_dn}/{sub_dir}/output'
      output = subprocess.check_output(shlex.split(cmd))
      output = output.decode("utf-8").split('\n')[:-1]
      infiles = ' '.join(output)
      cmd = f'hadd {root}{path_dn}/{sub_dir}/ntuple.root {infiles}'
      proc = subprocess.Popen(shlex.split(cmd),stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      (stdout, stderr) = proc.communicate()
      debug_hadd(stderr)
      print()
