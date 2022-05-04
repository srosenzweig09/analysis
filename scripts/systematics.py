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
parser.add_argument('--spherical', dest='spherical', help='', action='store_true', default=False)
parser.add_argument('--bias', dest='bias', help='', action='store_true', default=False)
parser.add_argument('--btag', dest='btag', help='', action='store_true', default=False)
args = parser.parse_args()

if args.rectangular: 
   cfg = 'config/rectConfig.cfg'
   region_type = 'rect'
if args.spherical: 
   cfg = 'config/sphereConfig.cfg'
   region_type = 'sphere'

def getROOTCanvas(h_title, bin_values, outFile):
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

config = ConfigParser()
config.optionxform = str
config.read(cfg)

minMX = int(config['plot']['minMX'])
maxMX = int(config['plot']['maxMX'])
nbins = int(config['plot']['nbins'])
mBins = np.linspace(minMX,maxMX,nbins)

root = 'root://cmseos.fnal.gov/'
if args.bias: 
   tag = 'dHHH_pairs'
   outdir = 'bias'
elif args.btag : 
   tag = 'dHHH_pairs_maxbtag'
   outdir = 'btag'
else:
   raise "Please provide --btag or --bias!"
sys_dir = f'/store/user/srosenzw/sixb/sixb_ntuples/Summer2018UL/{tag}/NMSSM/'
cmd = f'eos {root} ls {sys_dir}syst/'
output = subprocess.check_output(shlex.split(cmd))
systematics = output.decode("utf-8").split('\n')[:-1]

cmd = f'eos {root} ls {sys_dir}'
output = subprocess.check_output(shlex.split(cmd))
samples = output.decode("utf-8").split('\n')[:-1]
samples = [f"{sample}/ntuple.root" for sample in samples if 'NMSSM' in sample]

def get_SR(tree, region):
   if region == 'rect': 
      tree.rectangular_region(config)
      return tree.SRhs_mask
   elif region == 'sphere': 
      tree.spherical_region(config)
      return tree.A_SRhs_mask

for i,sample in enumerate(samples):
   if i >= 8: continue
   print(f"[INFO] Process sample: {sample}")
   text = re.search('MX_.*/', sample).group()[:-1].split('_')
   mx = int(text[1])
   my = int(text[3])
   file = root + sys_dir + sample
   tree = Signal(file)
   sr_mask = get_SR(tree, region_type)
   X_m = tree.X_m[sr_mask]
   n,e = np.histogram(X_m.to_numpy(), bins=mBins)

   outDir = f"combine/dHHH/{outdir}/{region_type}/"
   outFile = outDir + f'MX_{mx}_MY_{my}'

   try: del canvas, ROOT_hist, ROOT_hist_up, ROOT_hist_down
   except: pass
   canvas = ROOT.TCanvas('c1','c1', 600, 600)
   canvas.SetFrameLineWidth(3)
   canvas.Draw()

   h_title = f"signal"
   ROOT_hist = ROOT.TH1D(h_title,";m_{X} [GeV];Events",nbins-1,array('d',list(mBins)))
   for i,(val) in enumerate(n):
      ROOT_hist.SetBinContent(i+1, val)
   ROOT_hist.Draw("hist")

   fout = ROOT.TFile(f"{outFile}.root","recreate")
   fout.cd()
   ROOT_hist.Write()

   for systematic in systematics:
      print(f"[INFO] Processing systematic: {systematic}")
      if systematic == 'JER': 
         systematic = 'JER/pt'
      file_up = root + sys_dir + f'syst/{systematic}/up/{sample}'
      file_down = root + sys_dir + f'syst/{systematic}/down/{sample}'
      tree_up = Signal(file_up)
      tree_down = Signal(file_down)

      sr_mask = get_SR(tree_up, region_type)
      # scale = tree_up.scale
      X_m = tree_up.X_m[sr_mask]
      n_up,e = np.histogram(X_m.to_numpy(), bins=mBins)
      # n_up = n_up * scale

      sr_mask = get_SR(tree_down, region_type)
      # scale = tree_up.scale
      X_m = tree_down.X_m[sr_mask]
      n_down,e = np.histogram(X_m.to_numpy(), bins=mBins)
      # n_down = n_down * scale

      if '/' in systematic: systematic = systematic.replace('/','_')
      up_title = f"signal_{systematic}Up"
      down_title = f"signal_{systematic}Down"

      ROOT_hist_up = ROOT.TH1D(up_title,";m_{X} [GeV];Events",nbins-1,array('d',list(mBins)))
      ROOT_hist_down = ROOT.TH1D(down_title,";m_{X} [GeV];Events",nbins-1,array('d',list(mBins)))

      for i,(val_up,val_down) in enumerate(zip(n_up,n_down)):
         ROOT_hist_up.SetBinContent(i+1, val_up)
         ROOT_hist_down.SetBinContent(i+1, val_down)

      ROOT_hist_up.Draw("hist")
      ROOT_hist_down.Draw("hist")
      
      ROOT_hist_up.Write()
      ROOT_hist_down.Write()

   ROOT.gStyle.SetOptStat(0)

   # canvas.Print(f"{outFile}.pdf)","Title:Signal Region");

   fout.Close()

   print(f"File saved to {outFile}.root")