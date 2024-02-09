from argparse import ArgumentParser
from array import array
from configparser import ConfigParser
import numpy as np
import re
import ROOT
ROOT.gROOT.SetBatch(True)
import shlex
import subprocess
import sys
from utils.analysis import Signal

parser = ArgumentParser(description='Command line parser of model options and tags')
parser.add_argument('--rectangular', dest='rectangular', help='', action='store_true', default=False)
parser.add_argument('--vr', dest='vr', help='', action='store_true', default=False)
parser.add_argument('--nominal', dest='nominal', help='', action='store_true', default=False)
args = parser.parse_args()

cfg = 'config/bdt_params.cfg'
method = 'concentric'

def writeSystHist(systematic, sample, variation, method):
   file = root + sys_dir + f'syst/{systematic}/{variation}/{sample}'
   tree = Signal(file)

   print(".. getting SR")
   mask = get_region(tree, method)
   # scale = tree_up.scale
   print(".. getting X_m")
   X_m = tree.X_m[mask]
   print(".. building np hist")
   n,e = np.histogram(X_m.to_numpy(), bins=mBins)
   n = n * tree.scale
   # n_up = n_up * scale

   if '/' in systematic: systematic = systematic.replace('/','_')
   print(".. setting title")
   h_title = f"signal_{systematic}{variation.capitalize()}"

   print(".. creating ROOT hist")
   ROOT_hist = ROOT.TH1D(h_title,";m_{X} [GeV];Events",nbins,array('d',list(mBins)))

   print(".. setting bin content")
   for i,val in enumerate(n):
      ROOT_hist.SetBinContent(i+1, val)

   print(".. writing histogram")
   ROOT_hist.Draw("hist")
   ROOT_hist.Write()

def writeNominalHist(root, sys_dir, sample):
   fname = root + sys_dir + sample
   tree = Signal(fname)
   mask = get_region(tree, method)
   X_m = tree.X_m[mask]
   n,e = np.histogram(X_m.to_numpy(), bins=mBins)
   n = n * tree.scale

   h_title = f"signal"
   ROOT_hist = ROOT.TH1D(h_title,";m_{X} [GeV];Events",nbins,array('d',list(mBins)))
   for i,(val) in enumerate(n):
      ROOT_hist.SetBinContent(i+1, val)

   ROOT_hist.Draw("hist")
   ROOT_hist.Write()

def get_region(tree, region):
   if method == 'rect': 
      tree.rectangular_region(config)

      if region == 'v_sr': return tree.V_SRhs_mask
      else: return tree.A_SRhs_mask

   if method == 'sphere': 
      tree.spherical_region(config)
      
      if region == 'v_sr': return tree.V_SRhs_mask
      else: return tree.A_SRhs_mask
      
config = ConfigParser()
config.optionxform = str
config.read(cfg)

minMX = int(config['plot']['minMX'])
maxMX = int(config['plot']['maxMX'])
if config['plot']['style'] == 'linspace':
   nedges = int(config['plot']['nedges'])
   mBins = np.linspace(minMX,maxMX,nedges)
if config['plot']['style'] == 'arange':
   step = int(config['plot']['steps'])
   mBins = np.arange(minMX,maxMX,step)
   nedges = len(mBins)
nbins = nedges - 1

root = 'root://cmseos.fnal.gov/'
jets = 'feyn'
tag = 'maxbtag_4b'

sys_dir = f'/store/user/srosenzw/sixb/ntuples/Summer2018UL/{tag}/NMSSM/'
cmd = f'eos {root} ls {sys_dir}syst'
output = subprocess.check_output(shlex.split(cmd))
systematics = output.decode("utf-8").split('\n')[:-1]
if systematics[-1] == '{systematic}': systematics = systematics[:-1]

cmd = f'eos {root} ls {sys_dir}'
output = subprocess.check_output(shlex.split(cmd))
samples = output.decode("utf-8").split('\n')[:-1]
samples = [f"{sample}/ntuple.root" for sample in samples if 'NMSSM' in sample]

# sys.exit()

# skip = False
for i,sample in enumerate(samples):
   # if 'NMSSM_XYH_YToHH_6b_MX_1000_MY_800' in sample: skip = False
   # if 'NMSSM_XYH_YToHH_6b_MX_700_MY_400' not in sample: continue
   print(f"[SAMPLE] Processing: {sample}")
   # sys.exit()
   # if skip: continue

   text = re.search('MX_.*/', sample).group()[:-1].split('_')
   mx = int(text[1])
   my = int(text[3])

   region = 'a_sr'
   if args.vr: region = 'v_sr'
   outDir = f"combine/{jets}_{method}/{region}"
   outFile = f'{outDir}/MX_{mx}_MY_{my}'

   # numbered to avoid annoying message about canvas with same name
   canvas = ROOT.TCanvas(f'c{i}',f'c{i}', 600, 600)
   canvas.SetFrameLineWidth(3)
   canvas.Draw()

   fout = ROOT.TFile(f"{outFile}.root","recreate")
   fout.cd()

   writeNominalHist(root, sys_dir, sample)
   if args.nominal: continue

   for systematic in systematics:
      print(f"[SYSTEMATIC] Processing: {systematic}")
      if systematic == 'JER': 
         systematic = 'JER/pt'
      
      print("VARIATION: up")
      writeSystHist(systematic, sample, 'up')
      print("VARIATION: down")
      writeSystHist(systematic, sample, 'down')

   ROOT.gStyle.SetOptStat(0)

   print(".. closing file")
   fout.Close()
   del canvas

   print(f"File saved to {outFile}.root\n")