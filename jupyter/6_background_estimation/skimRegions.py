print("[INFO] .. starting program")

from argparse import ArgumentParser
import ast
import awkward as ak
from configparser import ConfigParser
import itertools
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import sys
import uproot
# https://pypi.org/project/uproot-tree-utils/
from uproot_tree_utils import clone_tree
import vector

### ------------------------------------------------------------------------------------
## Implement command line parser

print(".. parsing command line arguments")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--cfg',       dest='cfg',       help='config file' , required=True )
parser.add_argument('--input',     dest='input',     help='input file' ,  required=True )
parser.add_argument('--output',    dest='output',    help='output file' , required=True )
parser.add_argument('--is-signal', dest='is_signal', help='mark as MC',   action='store_true', default=False)

args = parser.parse_args()

### ------------------------------------------------------------------------------------
## Implement config parser

print(".. parsing config file")

config = ConfigParser()
config.optionxform = str
config.read(args.cfg)

treename = config['file']['tree']

maxSR = float(config['mass']['maxSR'])
maxVR = float(config['mass']['maxVR'])
maxCR = float(config['mass']['maxCR'])
if maxCR == -1: maxCR = 9999

score = float(config['score']['threshold'])

# BDT parameters
Nestimators  = int(config['BDT']['Nestimators'])
learningRate = float(config['BDT']['learningRate'])
maxDepth     = int(config['BDT']['maxDepth'])
minLeaves    = int(config['BDT']['minLeaves'])
GBsubsample  = float(config['BDT']['GBsubsample'])
randomState  = int(config['BDT']['randomState'])

variables    = config['BDT']['variables'].split(", ")

### ------------------------------------------------------------------------------------
## build region masks

print(".. opening input file")
tree = uproot.open(f"{args.input}:{treename}")
nwtree = uproot.open(f"{args.input}:NormWeightTree")

mH = 125 # GeV

HX_m = tree['HX_m'].array(library='np')
HY1_m = tree['HY1_m'].array(library='np')
HY2_m = tree['HY2_m'].array(library='np')

print(".. generating mass regions")

SR_mask = (abs(HX_m - mH) <= maxSR) & (abs(HY1_m - mH) <= maxSR) & (abs(HY2_m - mH) <= maxSR)
VR_mask = (abs(HX_m - mH) <= maxVR) & (abs(HY1_m - mH) <= maxVR) & (abs(HY2_m - mH) <= maxVR) & (abs(HX_m - mH) > maxSR) & (abs(HY1_m - mH) > maxSR) & (abs(HY2_m - mH) > maxSR)
CR_mask = (abs(HX_m - mH) <= maxCR) & (abs(HY1_m - mH) <= maxCR) & (abs(HY2_m - mH) <= maxCR) & (abs(HX_m - mH) > maxVR) & (abs(HY1_m - mH) > maxVR) & (abs(HY2_m - mH) > maxVR)

HX_b1_btag  = tree['HX_b1_DeepJet'].array(library='np')
HX_b2_btag  = tree['HX_b2_DeepJet'].array(library='np')
HY1_b1_btag = tree['HY1_b1_DeepJet'].array(library='np')
HY1_b2_btag = tree['HY1_b2_DeepJet'].array(library='np')
HY2_b1_btag = tree['HY2_b1_DeepJet'].array(library='np')
HY2_b2_btag = tree['HY2_b2_DeepJet'].array(library='np')

print(".. generating score regions")

btagavg = (HX_b1_btag + HX_b2_btag + HY1_b1_btag + HY1_b2_btag + HY2_b1_btag + HY2_b2_btag)/6

low_btag_mask  = btagavg < score
high_btag_mask = btagavg >= score

print(".. combining mass and score regions")

CR_ls_mask = ak.from_numpy(CR_mask & low_btag_mask)
CR_hs_mask = ak.from_numpy(CR_mask & high_btag_mask)

VR_ls_mask = ak.from_numpy(VR_mask & low_btag_mask)
VR_hs_mask = ak.from_numpy(VR_mask & high_btag_mask)

SR_ls_mask = ak.from_numpy(SR_mask & low_btag_mask)
if args.is_signal: SR_hs_mask = ak.from_numpy(SR_mask & high_btag_mask)
else: SR_hs_mask = ak.zeros_like(SR_ls_mask)

### ------------------------------------------------------------------------------------
## train BDT

print(".. preparing inputs to train BDT")

from utils.analysis import build_p4

HX_b1 = build_p4(
    tree['HX_b1_pt'].array(),
    tree['HX_b1_eta'].array(),
    tree['HX_b1_phi'].array(),
    tree['HX_b1_m'].array()
)
HX_b2 = build_p4(
    tree['HX_b2_pt'].array(),
    tree['HX_b2_eta'].array(),
    tree['HX_b2_phi'].array(),
    tree['HX_b2_m'].array()
)
HY1_b1 = build_p4(
    tree['HY1_b1_pt'].array(),
    tree['HY1_b1_eta'].array(),
    tree['HY1_b1_phi'].array(),
    tree['HY1_b1_m'].array()
)
HY1_b2 = build_p4(
    tree['HY1_b2_pt'].array(),
    tree['HY1_b2_eta'].array(),
    tree['HY1_b2_phi'].array(),
    tree['HY1_b2_m'].array()
)
HY2_b1 = build_p4(
    tree['HY2_b1_pt'].array(),
    tree['HY2_b1_eta'].array(),
    tree['HY2_b1_phi'].array(),
    tree['HY2_b1_m'].array()
)
HY2_b2 = build_p4(
    tree['HY2_b2_pt'].array(),
    tree['HY2_b2_eta'].array(),
    tree['HY2_b2_phi'].array(),
    tree['HY2_b2_m'].array()
)

HX_dr = HX_b1.deltaR(HX_b2)
HY1_dr = HY1_b1.deltaR(HY1_b2)
HY2_dr = HY2_b1.deltaR(HY2_b2)

vars = {'HX_dr':HX_dr, 'HY1_dr':HY1_dr, 'HY2_dr':HY2_dr}

features = {}
if ~args.is_signal:
    for var in variables:
        print(var)
        if var in tree.keys(): features[var] = tree[var].array()
        else: features[var] = vars[var]




sys.exit()

### ------------------------------------------------------------------------------------
## add branches and prepare to save

print(".. appending region masks to tree")

new_branches = {'CR_ls':CR_ls_mask, 'CR_hs':CR_hs_mask,'VR_ls':VR_ls_mask, 'VR_hs':VR_hs_mask, 'SR_ls':SR_ls_mask, 'SR_hs':SR_hs_mask}

print(".. adding new branches")

# newtree = {k:v for k,v in itertools.chain(branches,new_branches.items())}
# newtree = {k:v.array() for k,v in itertools.chain(tree.items(),new_branches.items())}
newtree = {}
for k,v in itertools.chain(tree.items(),new_branches.items()):
    try: newtree[k] = v.array()
    except: newtree[k] = v

NormWeightTree = {}
for k,v in nwtree.items():
    NormWeightTree[k] = v.array()

print(".. saving to output")
with uproot.recreate(args.output) as file:
    file[treename] = newtree
    file['NormWeightTree'] = NormWeightTree
    print(file[treename].show())