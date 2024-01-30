print(".. initiating script")
print(".. importing libraries")

from argparse import ArgumentParser
import awkward0 as ak0
import awkward as ak
from configparser import ConfigParser
# import pyhf
# pyhf.set_backend("jax")
# from pyhf.exceptions import FailedMinimization
import re
import uproot as up
import subprocess, shlex

# from array import array
# import ROOT
# ROOT.gROOT.SetBatch(True)
from utils.xsecUtils import lumiMap
# from utils.analysis.particle import Particle
import sys

from utils.analysis.signal import GNNSelection, DataTraining

print(".. parsing arguments")
parser = ArgumentParser()

parser.add_argument('--avg-btag-cut', dest='b_cut', default=0.59, type=float)
parser.add_argument('--avg-gnn-cut', dest='gnn_cut', default=0.75, type=float)
parser.add_argument('--train-data', dest='train_data', default=False, action='store_true')

args = parser.parse_args()

cfg = 'config/bias_config.cfg'

config = ConfigParser()
config.optionxform = str
config.read(cfg)

if args.train_data:
    data = DataTraining(config)

eos_base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/presel'
cmd = f'ls {eos_base}'
output = subprocess.run(shlex.split(cmd), capture_output=True).stdout
output = output.decode('UTF-8')
filelist = output.split('\n')

skip_list = ['tar', '10M', '100k']
if '' in filelist: filelist.remove('')

limits_dict = {}
files = []
for mass_file in filelist:
    ntuple = 'ntuple.root'
    if 'MX_700_MY_400' not in mass_file: continue
    if '_2M' in mass_file:
        ntuple = 'ntuple_test.root'

    skip_bool = any(substring in mass_file for substring in skip_list)
    if skip_bool: continue

    tree = GNNSelection(mass_file)
    print(tree.exp_limit)

    sys.exit()


    break


