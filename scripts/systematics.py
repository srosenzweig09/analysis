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
parser.add_argument('--spherical', dest='spherical', help='', action='store_true', default=True)
parser.add_argument('--bias', dest='bias', help='', action='store_true', default=False)
parser.add_argument('--btag', dest='btag', help='', action='store_true', default=False)
args = parser.parse_args()

if args.rectangular: 
   cfg = 'config/rectConfig.cfg'
   region_type = 'rect'
   args.spherical = False
if args.spherical: 
   cfg = 'config/sphereConfig.cfg'
   region_type = 'sphere'

def getROOTCanvas(h_title, bin_values, outFile, scale=1):
    print(f".. generating root hist for {h_title}")
    canvas = ROOT.TCanvas('c1','c1', 600, 600)
    canvas.SetFrameLineWidth(3)

    ROOT_hist = ROOT.TH1D(h_title,";m_{X} [GeV];Events",len(bin_values),array('d',list(mBins)))

    for i,(bin_vals) in enumerate(bin_values):
        ROOT_hist.SetBinContent(i+1, bin_vals)

    ROOT_hist.Draw("hist")
    canvas.Draw()
    ROOT.gStyle.SetOptStat(0)

    canvas.Print(f"{outFile}.pdf)","Title:Signal Region");

    fout = ROOT.TFile(f"{outFile}.root","recreate")
    fout.cd()
    ROOT_hist.Write()
    fout.Close()

def writeSystHist(systematic, sample, variation, region_type='sphere'):
   file = root + sys_dir + f'syst/{systematic}/{variation}/{sample}'
   tree = Signal(file)

   print(".. getting SR")
   sr_mask = get_SR(tree, region_type)
   # scale = tree_up.scale
   print(".. getting X_m")
   X_m = tree.X_m[sr_mask]
   print(".. building np hist")
   n,e = np.histogram(X_m.to_numpy(), bins=mBins)
   n = n * tree.scale
   # n_up = n_up * scale

   if '/' in systematic: systematic = systematic.replace('/','_')
   print(".. setting title")
   h_title = f"signal_{systematic}{variation.capitalize()}"

   print(".. creating ROOT hist")
   ROOT_hist = ROOT.TH1D(h_title,";m_{X} [GeV];Events",nbins-1,array('d',list(mBins)))

   print(".. setting bin content")
   for i,val in enumerate(n):
      ROOT_hist.SetBinContent(i+1, val)

   print(".. writing histogram")
   ROOT_hist.Draw("hist")
   ROOT_hist.Write()

def writeNominalHist(root, sys_dir, sample):
   file = root + sys_dir + sample
   tree = Signal(file)
   sr_mask = get_SR(tree, region_type)
   X_m = tree.X_m[sr_mask]
   n,e = np.histogram(X_m.to_numpy(), bins=mBins)
   n = n * tree.scale

   h_title = f"signal"
   ROOT_hist = ROOT.TH1D(h_title,";m_{X} [GeV];Events",nbins-1,array('d',list(mBins)))
   for i,(val) in enumerate(n):
      ROOT_hist.SetBinContent(i+1, val)

   ROOT_hist.Draw("hist")
   ROOT_hist.Write()

def get_SR(tree, region):
   if region == 'rect': 
      tree.rectangular_region(config)
      return tree.SRhs_mask
   elif region == 'sphere': 
      tree.spherical_region(config)
      return tree.A_SRhs_mask
      
config = ConfigParser()
config.optionxform = str
config.read(cfg)

minMX = int(config['plot']['minMX'])
maxMX = int(config['plot']['maxMX'])
nbins = int(config['plot']['nbins'])
mBins = np.linspace(minMX,maxMX,nbins)

root = 'root://cmseos.fnal.gov/'
if args.bias: 
   jets = 'bias'
   tag = 'dHHH_pairs'
elif args.btag : 
   jets = 'btag'
   tag = 'dHHH_pairs_maxbtag'
else:
   raise "Please provide --btag or --bias!"
sys_dir = f'/store/user/srosenzw/sixb/sixb_ntuples/Summer2018UL/{tag}/NMSSM/'
cmd = f'eos {root} ls {sys_dir}syst/'
output = subprocess.check_output(shlex.split(cmd))
systematics = output.decode("utf-8").split('\n')[:-1]
if systematics[-1] == '{systematic}': systematics = systematics[:-1]

cmd = f'eos {root} ls {sys_dir}'
output = subprocess.check_output(shlex.split(cmd))
samples = output.decode("utf-8").split('\n')[:-1]
samples = [f"{sample}/ntuple.root" for sample in samples if 'NMSSM' in sample]

# skip = False
for i,sample in enumerate(samples):
   print(f"[SAMPLE] Processing: {sample}")
   # if 'NMSSM_XYH_YToHH_6b_MX_1000_MY_800' in sample: skip = False
   # if skip: continue

   text = re.search('MX_.*/', sample).group()[:-1].split('_')
   mx = int(text[1])
   my = int(text[3])

   outDir = f"combine/{jets}_{region_type}"
   outFile = f'{outDir}/MX_{mx}_MY_{my}'

   canvas = ROOT.TCanvas('c1','c1', 600, 600)
   canvas.SetFrameLineWidth(3)
   canvas.Draw()

   fout = ROOT.TFile(f"{outFile}.root","recreate")
   fout.cd()

   writeNominalHist(root, sys_dir, sample)

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