from argparse import ArgumentParser
import subprocess, shlex

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--rIn', dest='rIn', default=None, required=True)
parser.add_argument('--rOut', dest='rOut', default=None, required=True)

args = parser.parse_args()

with open("/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/config/sphereConfig.cfg", "r") as f:
    lines = f.read().split('\n')
for i,line in enumerate(lines):
    if line == '[spherical]':
        index = i
        break

lines[i+2] = ' '.join(lines[i+2].split(' ')[:-1]) + f' {args.rIn}'
lines[i+3] = ' '.join(lines[i+3].split(' ')[:-1]) + f' {args.rOut}'

with open("/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/config/sphereConfig_new.cfg", "w") as f:
    f.write('\n'.join(lines))