"""
Author: Suzanne Rosenzweig
"""

import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from utils.useCMSstyle import *
plt.style.use(CMS)
from utils.plotter import Hist
from utils.analysis.particle import Particle, Higgs, Y
from utils.analysis.feyn import model_path, model_name
from utils.xsecUtils import lumiMap, xsecMap
from utils.plotter import latexTitle

from rich.console import Console
console = Console()
import re
import os, sys 
from colorama import Fore, Style
# import subprocess

njet_bins = np.arange(8)
id_bins = np.arange(-1, 7)
pt_bins = np.linspace(0, 500, 100)
score_bins = np.linspace(0,1.01,102)

nbins = 40
m_bins = np.linspace(375, 1150, nbins)
x_X_m = (m_bins[:-1] + m_bins[1:]) / 2

def get_region_mask(higgs, center, sr_edge, cr_edge):
   deltaM = np.column_stack(([abs(mH.to_numpy() - val) for mH,val in zip(higgs,center)]))
   deltaM = deltaM * deltaM
   deltaM = deltaM.sum(axis=1)
   deltaM = np.sqrt(deltaM)

   sr_mask = deltaM <= sr_edge 
   cr_mask = (deltaM > sr_edge) & (deltaM <= cr_edge) 

   return sr_mask, cr_mask

def get_hs_ls_masks(sr_mask, cr_mask, ls_mask, hs_mask):
   cr_ls_mask = cr_mask & ls_mask
   cr_hs_mask = cr_mask & hs_mask
   sr_ls_mask = sr_mask & ls_mask
   sr_hs_mask = sr_mask & hs_mask

   return cr_ls_mask, cr_hs_mask, sr_ls_mask, sr_hs_mask

def ROOTHist(h_vals, title, filename):
   """
   title : should be either 'signal' or 'data'
   """
   import ROOT
   ROOT.gROOT.SetBatch(True)
   from array import array
   assert ".root" in filename, print("[ERROR] Please include '.root' in filename")
   
   fout = ROOT.TFile(f"{filename}","recreate")
   fout.cd()

   canvas = ROOT.TCanvas('c1','c1', 600, 600)
   canvas.SetFrameLineWidth(3)
   canvas.Draw()

   h_title = title
   ROOT_hist = ROOT.TH1D(h_title,";m_{X} [GeV];Events",nbins-1,array('d',list(m_bins)))
   for i,(val) in enumerate(h_vals):
      ROOT_hist.SetBinContent(i+1, val) 

   ROOT_hist.Draw("hist")
   ROOT_hist.Write()
   fout.Close()

from configparser import ConfigParser
# Parent Class
class Tree():

   def __init__(self, filename, treename='sixBtree', cfg='config/bdt_params.cfg', feyn=True):
      from utils.cutConfig import btagWP
      self.btagWP = btagWP

      if isinstance(cfg, str):
         config = ConfigParser()
         config.optionxform = str
         config.read(cfg)
         self.config = config
      elif isinstance(cfg, ConfigParser):
         self.config = cfg

      self.filename = filename
      self.treename = treename
      self.year = int([yr for yr in ['2016', '2017', '2018'] if yr in filename][0])
      self.is_signal = 'NMSSM' in filename

      file_info = '/'.join(self.filename.split('/')[8:])
      # print(file_info)
      # console.log(f"[cyan]Loading {file_info}...")
      print(f"{Fore.CYAN}ntuple: {self.filename}{Style.RESET_ALL}")

      self.tree = uproot.open(f"{filename}:{treename}")

      # with uproot.open(f"{filename}:{treename}") as tree:
      pattern = re.compile('H.+_') # Search for any keys beginning with an 'H' and followed somewhere by a '_'
      for k, v in self.tree.items():
         if re.match(pattern, k) or 'jet' in k or 'gen' in k or 'Y' in k or 'X' in k:
            setattr(self, k, v.array())
      try: 
         with uproot.open(f"{filename}:h_cutflow_unweighted") as f: cutflow = f
      except:
         with uproot.open(f"{filename}:h_cutflow") as f: cutflow = f
      self.cutflow_labels = cutflow.axis().labels()
      # self.tree = tree
      self.nevents = int(cutflow.to_numpy()[0][-1])
      self.total = int(cutflow.to_numpy()[0][0])
      self.scale = 1

      self.cutflow = (cutflow.to_numpy()[0]).astype(int)
      self.cutflow_norm = (cutflow.to_numpy()[0]/cutflow.to_numpy()[0][0]*100).astype(int)

      key = str(self.year)
      if key == '2016' and 'preVFP' in self.filename: key += 'preVFP'
      elif key == '2016': key += 'postVFP'
      self.year_key = key
      self.loose_wp = self.btagWP[key]['Loose']
      self.medium_wp = self.btagWP[key]['Medium']
      self.tight_wp = self.btagWP[key]['Tight']

      if feyn: self.init_model(feyn)
      else: self.initialize_bosons()
      
      self.initialize_vars()
      if not self.is_signal:
         self.set_var_dict()
      
      if self.nevents != len(self.n_jet): print("WARNING! Number of events in cutflow does not match number of events in tree!")

   def keys(self):
      return uproot.open(f"{self.filename}:{self.treename}").keys()

   def find_key(self, start):
      for key in self.keys():
         if start in key : print(key)

   def get(self, key, library='ak'):
      """Returns the key.
      Use case: Sometimes keys are accessed directly, e.g. tree.key, but other times, the key may be accessed from a list of keys, e.g. tree.get(['key']). This function handles the latter case.
      """
      try: 
         arr = getattr(self, key)
         if library=='np' and not isinstance(arr, np.ndarray): arr = arr.to_numpy()
         return arr
      except:
         return self.tree[key].array(library=library)
    
   def np(self, key):
      """Returns the key as a numpy array."""
      np_arr = self.get(key, library='np')
      if not isinstance(np_arr, np.ndarray): np_arr = np_arr.to_numpy()
      return np_arr


   def init_model(self, feyn):
      if self.is_signal:
         ### FIX ME FOR 2016/2017
         year = self.year
         if 'Summer2016UL' in self.filename and 'preVFP' in self.filename: year = self.year + 'preVFP'
         if 'Summer2016UL' in self.filename and 'preVFP' not in self.filename: year = self.year + 'postVFP'
         # print(updated_model_path)

         if isinstance(feyn, bool): 
            feyn_path = self.filename[self.filename.find('NMSSM/')+6:].replace('/ntuple','')
            self.feyn = f"{model_path}/{year}/{feyn_path}"
            if 'maxbtag/' in self.filename: self.feyn = f"{model_path}/{self.year}/maxbtag/{feyn_path}"
         else: self.feyn = feyn
      else:
         data_name = [entry for entry in self.filename.split('/') if 'Data' in entry][0]
         self.feyn = f"{model_path}/{self.year}/{data_name}.root"

      # console.log(f"[purple]Loading {self.feyn}...")
      print(f"{Fore.MAGENTA}model: {self.feyn}{Style.RESET_ALL}")

      # if 'awkd' in model:
      #    import awkward0 as ak0
      #    # with ak0.load(model) as f_awk:
      #    with ak0.load(self.feyn) as f_awk:
      #       try: self.scores = ak.unflatten(f_awk['scores'], np.repeat(45, self.nevents)).to_numpy()
      #       except: self.scores = ak.from_regular(f_awk['scores'].astype(float))
      #       self.nmaxscore = self.scores[ak.argsort(self.scores, ascending=False)][:,1]
      #       self.maxscore = f_awk['maxscore']
      #       self.max_diff = self.maxscore - self.nmaxscore
      #       assert np.array_equal(self.maxscore, ak.max(self.scores, axis=1))
      #       self.minscore = ak.min(self.scores, axis=1)
      #       self.maxcomb = f_awk['max_comb']
      #       self.maxlabel = f_awk['max_label']
      #       self.mass_rank = np.concatenate(f_awk['mass_rank'])
      #       self.nres_rank = np.concatenate(f_awk['nres_rank'])
      # else:
      with uproot.open(self.feyn) as f:
         f = f['Events']
         self.scores = f['scores'].array(library='np')
         self.maxcomb = f['max_comb'].array(library='np')
         self.maxscore = f['max_score'].array()
         self.maxlabel = f['max_label'].array()
         self.minscore = f['min_score'].array()
         self.nres_rank = f['nres_rank'].array()
         self.mass_rank = f['mass_rank'].array()
         self.max_diff = np.sort(self.scores, axis=1)[:,44]-np.sort(self.scores, axis=1)[:,43]

      combos = self.maxcomb.astype(int)
      combos = ak.from_regular(combos)

      self.combos = combos

      btag_mask = ak.argsort(self.jet_btag, axis=1, ascending=False) < 6

      pt = self.jet_ptRegressed[btag_mask][combos]
      phi = self.jet_phi[btag_mask][combos]
      eta = self.jet_eta[btag_mask][combos]
      m = self.jet_mRegressed[btag_mask][combos]
      btag = self.jet_btag[btag_mask][combos]
      sig_id = self.jet_signalId[btag_mask][combos]
      h_id = (self.jet_signalId[btag_mask][combos] + 2) // 2

      self.btag_avg = ak.mean(btag, axis=1)

      sample_particles = []
      for j in range(6):
         particle = Particle({
               'pt' : pt[:,j],
               'eta' : eta[:,j],
               'phi' : phi[:,j],
               'm' : m[:,j],
               'btag' : btag[:,j],
               'sig_id' : sig_id[:,j],
               'h_id': h_id[:,j]
               }
         )
         sample_particles.append(particle)

      HX_b1 = {'pt':sample_particles[0].pt,'eta':sample_particles[0].eta,'phi':sample_particles[0].phi,'m':sample_particles[0].m,'btag':sample_particles[0].btag,'sig_id':sample_particles[0].sig_id,'h_id':sample_particles[0].h_id}
      HX_b2 = {'pt':sample_particles[1].pt,'eta':sample_particles[1].eta,'phi':sample_particles[1].phi,'m':sample_particles[1].m,'btag':sample_particles[1].btag,'sig_id':sample_particles[1].sig_id,'h_id':sample_particles[1].h_id}
      H1_b1 = {'pt':sample_particles[2].pt,'eta':sample_particles[2].eta,'phi':sample_particles[2].phi,'m':sample_particles[2].m,'btag':sample_particles[2].btag,'sig_id':sample_particles[2].sig_id,'h_id':sample_particles[2].h_id}
      H1_b2 = {'pt':sample_particles[3].pt,'eta':sample_particles[3].eta,'phi':sample_particles[3].phi,'m':sample_particles[3].m,'btag':sample_particles[3].btag,'sig_id':sample_particles[3].sig_id,'h_id':sample_particles[3].h_id}
      H2_b1 = {'pt':sample_particles[4].pt,'eta':sample_particles[4].eta,'phi':sample_particles[4].phi,'m':sample_particles[4].m,'btag':sample_particles[4].btag,'sig_id':sample_particles[4].sig_id,'h_id':sample_particles[4].h_id}
      H2_b2 = {'pt':sample_particles[5].pt,'eta':sample_particles[5].eta,'phi':sample_particles[5].phi,'m':sample_particles[5].m,'btag':sample_particles[5].btag,'sig_id':sample_particles[5].sig_id,'h_id':sample_particles[5].h_id}

      self.HX = Higgs(HX_b1, HX_b2)

      # self.HX_b1 = self.HX.b1
      # self.HX_b2 = self.HX.b2

      H1 = Higgs(H1_b1, H1_b2)
      H2 = Higgs(H2_b1, H2_b2)

      assert ak.all(self.HX.b1.pt >= self.HX.b2.pt)
      assert ak.all(H1.b1.pt >= H1.b2.pt)
      assert ak.all(H2.b1.pt >= H2.b2.pt)

      self.Y = Y(H1, H2)

      self.H1 = self.Y.H1
      self.H2 = self.Y.H2

      assert ak.all(self.H1.pt >= self.H2.pt)

      self.X = self.HX + self.H1 + self.H2

      self.H_b_sig_id = np.column_stack((
         self.HX.b1.sig_id.to_numpy(),
         self.HX.b2.sig_id.to_numpy(),
         self.H1.b1.sig_id.to_numpy(),
         self.H1.b2.sig_id.to_numpy(),
         self.H2.b1.sig_id.to_numpy(),
         self.H2.b2.sig_id.to_numpy(),
      ))

      self.H_b_h_id = np.column_stack((
         self.HX.b1.h_id.to_numpy(),
         self.HX.b2.h_id.to_numpy(),
         self.H1.b1.h_id.to_numpy(),
         self.H1.b2.h_id.to_numpy(),
         self.H2.b1.h_id.to_numpy(),
         self.H2.b2.h_id.to_numpy(),
      ))

      self.gnn_resolved_mask = ak.all(self.H_b_sig_id > -1, axis=1)
      self.gnn_resolved_h_mask = ak.all(self.H_b_h_id > 0, axis=1)

      self.HX_correct = (self.HX.b1.h_id == self.HX.b2.h_id) & (self.HX.b1.h_id == 1)
      self.H1_correct = (self.H1.b1.h_id == self.H1.b2.h_id) & ((self.H1.b1.h_id == 2) | (self.H1.b1.h_id == 3))
      self.H2_correct = (self.H2.b1.h_id == self.H2.b2.h_id) & ((self.H2.b1.h_id == 2) | (self.H2.b1.h_id == 3))

      # print(self.H2_correct)


      # self.HX_correct = (self.HX.b1.h_id == self.HX.b2.h_id) & (self.HX.b1.h_id == 2)
      # self.H1_correct = (self.H1.b1.h_id == self.H1.b2.h_id) & ((self.H1.b1.h_id == 1) | (self.H1.b1.h_id == 0))
      # self.H2_correct = (self.H2.b1.h_id == self.H2.b2.h_id) & ((self.H2.b1.h_id == 0) | (self.H2.b1.h_id == 1))

      # efficiency taking into account to which higgs the pair was assigned
      self.n_H_correct = self.HX_correct*1 + self.H1_correct*1 + self.H2_correct*1

      self.HA_correct = (self.HX.b1.h_id == self.HX.b2.h_id) & (self.HX.b1.h_id > 0)
      self.HB_correct = (self.H1.b1.h_id == self.H1.b2.h_id) & (self.H1.b1.h_id > 0)
      self.HC_correct = (self.H2.b1.h_id == self.H2.b2.h_id) & (self.H2.b1.h_id > 0)

      # efficiency without taking into account to which higgs the pair was assigned
      # only worried about pairing together two jets from any higgs boson
      self.n_H_paired_correct = self.HA_correct*1 + self.HB_correct*1 + self.HC_correct*1


   def initialize_bosons(self):
      self.HX_b1 = Particle(self, 'HX_b1')
      self.HX_b2 = Particle(self, 'HX_b2')
      self.H1_b1 = Particle(self, 'H1_b1')
      self.H1_b2 = Particle(self, 'H1_b2')
      self.H2_b1 = Particle(self, 'H2_b1')
      self.H2_b2 = Particle(self, 'H2_b2')

      self.HX = Particle(self, 'HX')
      self.H1 = Particle(self, 'H1')
      self.H2 = Particle(self, 'H2')

      setattr(self.HX, 'b1', self.HX_b1)
      setattr(self.HX, 'b2', self.HX_b2)
      setattr(self.H1, 'b1', self.H1_b1)
      setattr(self.H1, 'b2', self.H1_b2)
      setattr(self.H2, 'b1', self.H2_b1)
      setattr(self.H2, 'b2', self.H2_b2)

      setattr(self.HX, 'dr', self.HX.b1.deltaR(self.HX.b2))
      setattr(self.H1, 'dr', self.H1.b1.deltaR(self.H1.b2))
      setattr(self.H2, 'dr', self.H2.b1.deltaR(self.H2.b2))

      self.Y = Particle(self, 'Y')

      self.X = Particle(self, 'X')

      self.HX_dr = self.HX_b1.deltaR(self.HX_b2)
      self.H1_dr = self.H1_b1.deltaR(self.H1_b2)
      self.H2_dr = self.H2_b1.deltaR(self.H2_b2)

      self.HX_m = self.HX.m
      self.H1_m = self.H1.m
      self.H2_m = self.H2.m

      self.HX_H1_dEta = self.HX.deltaEta(self.H1)
      self.H1_H2_dEta = self.H1.deltaEta(self.H2)
      self.H2_HX_dEta = self.H2.deltaEta(self.HX)

      self.HX_H1_dPhi = self.HX.deltaPhi(self.H1)
      self.H1_H2_dPhi = self.H1.deltaPhi(self.H2)
      self.H2_HX_dPhi = self.H2.deltaPhi(self.HX)

      self.HX_costheta = self.HX.costheta
      self.H1_costheta = self.H1.costheta
      self.H2_costheta = self.H2.costheta

   def initialize_vars(self):
      """Initialize variables that don't exist in the original ROOT tree."""

      self.tight_mask = self.jet_btag > self.tight_wp
      medium_mask = self.jet_btag > self.medium_wp
      loose_mask = self.jet_btag > self.loose_wp

      self.fail_mask = ~loose_mask
      self.loose_mask = loose_mask & ~medium_mask
      self.medium_mask = medium_mask & ~self.tight_mask

      self.n_tight = ak.sum(self.tight_mask, axis=1)
      self.n_medium = ak.sum(self.medium_mask, axis=1)
      self.n_loose = ak.sum(self.loose_mask, axis=1)
      self.n_fail = ak.sum(self.fail_mask, axis=1)

      bs = [self.HX.b1, self.HX.b2, self.H1.b1, self.H1.b2, self.H2.b1, self.H2.b2]
      pair1 = [self.HX.b1]*5 + [self.HX.b2]*4 + [self.H1.b1]*3 + [self.H1.b2]*2 + [self.H2.b1]
      pair2 = bs[1:] + bs[2:] + bs[3:] + bs[4:] + [bs[-1]]

      dR6b = []
      dEta6b = []
      for b1, b2 in zip(pair1, pair2):
         dR6b.append(b1.deltaR(b2))
         dEta6b.append(abs(b1.deltaEta(b2)))
      dR6b = np.column_stack(dR6b)
      dEta6b = np.column_stack(dEta6b)
      self.dR6bmin = dR6b.min(axis=1)
      self.dEta6bmax = dEta6b.max(axis=1)

      self.pt6bsum = self.HX.b1.pt + self.HX.b2.pt + self.H1.b1.pt + self.H1.b2.pt + self.H2.b1.pt + self.H2.b2.pt

      self.HX_pt = self.HX.pt
      self.H1_pt = self.H1.pt
      self.H2_pt = self.H2.pt

      self.HX_dr = self.HX.dr
      self.H1_dr = self.H1.dr
      self.H2_dr = self.H2.dr

      self.HX_m = self.HX.m
      self.H1_m = self.H1.m
      self.H2_m = self.H2.m

      self.HX_H1_dEta = abs(self.HX.deltaEta(self.H1))
      self.H1_H2_dEta = abs(self.H1.deltaEta(self.H2))
      self.H2_HX_dEta = abs(self.H2.deltaEta(self.HX))

      self.HX_H1_dPhi = self.HX.deltaPhi(self.H1)
      self.H1_H2_dPhi = self.H1.deltaPhi(self.H2)
      self.H2_HX_dPhi = self.H2.deltaPhi(self.HX)

      self.HX_H1_dr = self.HX.deltaR(self.H1)
      self.HX_H2_dr = self.H2.deltaR(self.HX)
      self.H1_H2_dr = self.H1.deltaR(self.H2)
      
      self.Y_HX_dr = self.Y.deltaR(self.HX)

      self.HX_costheta = abs(np.cos(self.HX.P4.theta))
      self.H1_costheta = abs(np.cos(self.H1.P4.theta))
      self.H2_costheta = abs(np.cos(self.H2.P4.theta))

      self.HX_H1_dr = self.HX.deltaR(self.H1)
      self.H1_H2_dr = self.H2.deltaR(self.H1)
      self.H2_HX_dr = self.HX.deltaR(self.H2)

      self.X_m = self.X.m

      self.X_m_prime = self.X_m - self.HX_m - self.H1_m - self.H2_m + 3*125

      self.H_j_btag = np.column_stack((
         self.HX_b1_btag.to_numpy(),
         self.HX_b2_btag.to_numpy(),
         self.H1_b1_btag.to_numpy(),
         self.H1_b2_btag.to_numpy(),
         self.H2_b1_btag.to_numpy(),
         self.H2_b2_btag.to_numpy()
      ))

      self.n_H_jet_tight = np.sum(self.H_j_btag >= self.tight_wp, axis=1)
      self.n_H_jet_medium = np.sum(self.H_j_btag >= self.medium_wp, axis=1)
      self.n_H_jet_loose = np.sum(self.H_j_btag >= self.loose_wp, axis=1)

      self.btag_avg = np.average(self.H_j_btag, axis=1)

      # if self._is_signal and not self.gnn:
      #    self.HA_correct = (self.HX_b1.h_id == self.HX_b2.h_id) & (self.HX_b1.h_id != -1)
      #    self.HB_correct = (self.H1_b1.h_id == self.H1_b2.h_id) & (self.H1_b1.h_id != -1)
      #    self.HC_correct = (self.H2_b1.h_id == self.H2_b2.h_id) & (self.H2_b1.h_id != -1)

      #    self.n_H_paired_correct = self.HA_correct*1 + self.HB_correct*1 + self.HC_correct*1

      #    self.HX_correct = (self.HX_b1.h_id == self.HX_b2.h_id) & (self.HX_b1.h_id == 0)
      #    self.H1_correct = (self.H1_b1.h_id == self.H1_b2.h_id) & (self.H1_b1.h_id == 1)
      #    self.H2_correct = (self.H2_b1.h_id == self.H2_b2.h_id) & (self.H2_b1.h_id == 2)

      #    self.n_H_correct = self.HX_correct*1 + self.H1_correct*1 + self.H2_correct*1

   def initialize_gen(self):
      """Initializes gen-matched particles."""

      self.gen_matched_HX_b1 = Particle(self, 'gen_HX_b1')
      self.gen_matched_HX_b2 = Particle(self, 'gen_HX_b2')
      self.gen_matched_H1_b1 = Particle(self, 'gen_H1_b1')
      self.gen_matched_H1_b2 = Particle(self, 'gen_H1_b2')
      self.gen_matched_H2_b1 = Particle(self, 'gen_H2_b1')
      self.gen_matched_H2_b2 = Particle(self, 'gen_H2_b2')

      self.gen_matched_HX = self.gen_matched_HX_b1 + self.gen_matched_HX_b2
      self.gen_matched_H1 = self.gen_matched_H1_b1 + self.gen_matched_H1_b2
      self.gen_matched_H2 = self.gen_matched_H2_b1 + self.gen_matched_H2_b2

      good_mask = ak.nan_to_num(self.gen_matched_HX.m, nan=-1) > -1
      good_mask = good_mask & (ak.nan_to_num(self.gen_matched_H1.m, nan=-1) > -1)
      self.good_mask = good_mask & (ak.nan_to_num(self.gen_matched_H2.m, nan=-1) > -1)

      # self.resolved_mask = ak.all(self.jet_signalId[:,:6] > -1, axis=1)

   def diagonal_region(self):
      
      minMX = int(self.config['plot']['minMX'])
      maxMX = int(self.config['plot']['maxMX'])
      if self.config['plot']['style'] == 'linspace':
         nbins = int(self.config['plot']['edges'])
         self.mBins = np.linspace(minMX,maxMX,nbins)
      if self.config['plot']['style'] == 'arange':
         step = int(self.config['plot']['steps'])
         self.mBins = np.arange(minMX,maxMX,step)

      self.x_mBins = (self.mBins[:-1] + self.mBins[1:])/2

      """Defines spherical estimation region masks."""
      self.ar_center = float(self.config['spherical']['ARcenter'])
      self.sr_edge   = float(self.config['spherical']['SRedge'])
      self.cr_edge   = float(self.config['spherical']['CRedge'])

      self.vr_center = self.ar_center + (self.sr_edge + self.cr_edge) / 2

      # higgs = ['HX_m', 'H1_m', 'H2_m']
      higgs = [self.HX.m, self.H1.m, self.H2.m]

      # deltaM = np.column_stack(([getattr(self, mH).to_numpy() - self.ar_center for mH in higgs]))
      deltaM = np.column_stack(([abs(mH.to_numpy() - self.ar_center) for mH in higgs]))
      deltaM = deltaM * deltaM
      deltaM = deltaM.sum(axis=1)
      self.AR_deltaM = np.sqrt(deltaM)
      self.asr_mask = self.AR_deltaM <= self.sr_edge # Analysis SR
      if self._is_signal: self.asr_avgbtag = self.btag_avg[self.asr_mask]
      self.acr_mask = (self.AR_deltaM > self.sr_edge) & (self.AR_deltaM <= self.cr_edge) # Analysis CR
      if not self._is_signal: self.acr_avgbtag = self.btag_avg[self.acr_mask]

      # VR_deltaM = np.column_stack(([abs(getattr(self, mH).to_numpy() - self.vr_center) for mH in higgs]))
      VR_deltaM = np.column_stack(([abs(mH.to_numpy() - self.vr_center) for mH in higgs]))
      VR_deltaM = VR_deltaM * VR_deltaM
      VR_deltaM = VR_deltaM.sum(axis=1)
      VR_deltaM = np.sqrt(VR_deltaM)
      self.vsr_mask = VR_deltaM <= self.sr_edge # Validation SR
      self.vcr_mask = (VR_deltaM > self.sr_edge) & (VR_deltaM <= self.cr_edge) # Validation CR

      self.score_cut = float(self.config['score']['threshold'])
      self.ls_mask = self.btag_avg < self.score_cut # ls
      self.hs_mask = self.btag_avg >= self.score_cut # hs

      self.acr_ls_mask = self.acr_mask & self.ls_mask
      self.acr_hs_mask = self.acr_mask & self.hs_mask
      self.asr_ls_mask = self.asr_mask & self.ls_mask
      self.asr_hs_mask = self.asr_mask & self.hs_mask
      if not self._is_signal:
         self.blind_mask = ~self.asr_hs_mask
         self.asr_hs_mask = np.zeros_like(self.asr_mask)
      self.sr_mask = self.asr_hs_mask
      # else: self.acr_avgbtag = self.btag_avg[self.vcr_mask]

      self.vcr_ls_mask = self.vcr_mask & self.ls_mask
      self.vcr_hs_mask = self.vcr_mask & self.hs_mask
      self.vsr_ls_mask = self.vsr_mask & self.ls_mask
      self.vsr_hs_mask = self.vsr_mask & self.hs_mask

   def spherical_region(self, nregions='concentric'):
      # print(self.config['spherical']['nregions'])
      # if self.config['spherical']['nregions'] == 'multiple':
      if nregions == 'multiple':
         print("REGION: multiple")
         self.multi_region()
      # elif self.config['spherical']['nregions'] == 'diagonal':
      elif nregions == 'diagonal':
         print("REGION: diagonal")
         self.diagonal_region()
      # elif self.config['spherical']['nregions'] == 'concentric':
      elif nregions == 'concentric':
         print("REGION: concentric")
         self.concentric_spheres_region()

   def concentric_spheres_region(self):
      minMX = int(self.config['plot']['minMX'])
      maxMX = int(self.config['plot']['maxMX'])
      if self.config['plot']['style'] == 'linspace':
         nbins = int(self.config['plot']['edges'])
         self.mBins = np.linspace(minMX,maxMX,nbins)
      if self.config['plot']['style'] == 'arange':
         step = int(self.config['plot']['steps'])
         self.mBins = np.arange(minMX,maxMX,step)

      self.x_mBins = (self.mBins[:-1] + self.mBins[1:])/2

      """Defines spherical estimation region masks."""
      self.ar_center = float(self.config['spherical']['ARcenter'])
      self.sr_edge   = float(self.config['spherical']['SRedge'])
      self.vr_edge   = float(self.config['spherical']['VRedge'])
      self.cr_edge   = float(self.config['spherical']['CRedge'])

      # higgs = ['HX_m', 'H1_m', 'H2_m']
      higgs = [self.HX.m, self.H1.m, self.H2.m]
      try: higgs = [m.to_numpy() for m in higgs]
      except: pass

      # deltaM = np.column_stack(([getattr(self, mH).to_numpy() - self.ar_center for mH in higgs]))
      deltaM = np.column_stack(([abs(mH - self.ar_center) for mH in higgs]))
      deltaM = deltaM * deltaM
      deltaM = deltaM.sum(axis=1)
      deltaM = np.sqrt(deltaM)
      self.deltaM = deltaM

      self.asr_mask = deltaM <= self.sr_edge # Analysis SR
      if self._is_signal: self.asr_avgbtag = self.btag_avg[self.asr_mask]
      self.acr_mask = (deltaM > self.sr_edge) & (deltaM <= self.cr_edge) # Analysis CR
      if not self._is_signal: self.acr_avgbtag = self.btag_avg[self.acr_mask]

      # vsr_edge = self.sr_edge*2 + self.cr_edge
      # vcr_edge = self.sr_edge*2 + self.cr_edge*2
      self.vsr_mask = (deltaM <= self.cr_edge) &  (deltaM > self.sr_edge)# Validation SR
      self.vcr_mask = (deltaM > self.cr_edge) & (deltaM <= self.vr_edge) # Validation CR

      self.score_cut = float(self.config['score']['threshold'])
      self.ls_mask = self.btag_avg < self.score_cut # ls
      self.hs_mask = self.btag_avg >= self.score_cut # hs


      # b_cut = float(config['score']['n'])
      # self.nloose_b = ak.sum(self.get('jet_btag') > 0.0490, axis=1)
      # self.nmedium_b = ak.sum(self.get('jet_btag') > 0.2783, axis=1)
      # ls_mask = self.nmedium_b < b_cut # ls
      # hs_mask = self.nmedium_b >= b_cut # hs

      self.acr_ls_mask = self.acr_mask & self.ls_mask
      self.acr_hs_mask = self.acr_mask & self.hs_mask
      self.asr_ls_mask = self.asr_mask & self.ls_mask
      self.asr_hs_mask = self.asr_mask & self.hs_mask
      if not self._is_signal:
         self.blind_mask = ~self.asr_hs_mask
         self.asr_hs_mask = np.zeros_like(self.asr_mask)
      # else: self.acr_avgbtag = self.btag_avg[self.vcr_mask]

      self.vcr_ls_mask = self.vcr_mask & self.ls_mask
      self.vcr_hs_mask = self.vcr_mask & self.hs_mask
      self.vsr_ls_mask = self.vsr_mask & self.ls_mask
      self.vsr_hs_mask = self.vsr_mask & self.hs_mask
   
      self.crtf = round(1 + np.sqrt(1/self.acr_ls_mask.sum()+1/self.acr_hs_mask.sum()),2)
      self.vrtf = 1 + np.sqrt(1/self.vcr_ls_mask.sum()+1/self.vcr_hs_mask.sum())

   def multi_region(self):
      """
      The GNN tends to flatten out the 2D MH_i v. MH_j distribution, leaving fewer events in the validation region and posing a potential problem when it comes to obtaining closure. This function introduces more validation regions, which makes the validation plots look better... but now that I think about it, I'm changing the validation regions without affecting at all the signal region and estimation going on in there... How can I show with confidence that whatever validation region I use is a good representative of the signal region?

      Ultimately the goal is to apply the background modeling procedure in the validation region, show that it works, and obtain confidence that it will work in the signal region... but we should be able to show that the validation region is kinematically similar to the signal region, right? I'm not quite sure anymore.
      """

      minMX = int(self.config['plot']['minMX'])
      maxMX = int(self.config['plot']['maxMX'])
      if self.config['plot']['style'] == 'linspace':
         nbins = int(self.config['plot']['edges'])
         self.mBins = np.linspace(minMX,maxMX,nbins)
      if self.config['plot']['style'] == 'arange':
         step = int(self.config['plot']['steps'])
         self.mBins = np.arange(minMX,maxMX,step)

      self.x_mBins = (self.mBins[:-1] + self.mBins[1:])/2

      """Defines spherical estimation region masks."""
      self.ar_center = float(self.config['spherical']['ARcenter'])
      self.sr_edge   = float(self.config['spherical']['SRedge'])
      self.cr_edge   = float(self.config['spherical']['CRedge'])

      deltaS = (self.cr_edge + self.sr_edge) / np.sqrt(2)
      
      val_centers = []
      val_centers.append((self.ar_center + deltaS, self.ar_center + deltaS, self.ar_center + deltaS))
      val_centers.append((self.ar_center + deltaS, self.ar_center + deltaS, self.ar_center - deltaS))
      val_centers.append((self.ar_center + deltaS, self.ar_center - deltaS, self.ar_center + deltaS))
      val_centers.append((self.ar_center - deltaS, self.ar_center + deltaS, self.ar_center + deltaS))
      val_centers.append((self.ar_center + deltaS, self.ar_center - deltaS, self.ar_center - deltaS))
      val_centers.append((self.ar_center - deltaS, self.ar_center + deltaS, self.ar_center - deltaS))
      val_centers.append((self.ar_center - deltaS, self.ar_center - deltaS, self.ar_center + deltaS))
      val_centers.append((self.ar_center - deltaS, self.ar_center - deltaS, self.ar_center - deltaS))

      higgs = [self.HX.m, self.H1.m, self.H2.m]

      self.asr_mask, self.acr_mask = get_region_mask(higgs, (125,125,125), self.sr_edge, self.cr_edge)

      vsr_masks, vcr_masks = [], []
      self.vsr_mask = np.repeat(False, len(self.asr_mask))
      self.vcr_mask = np.repeat(False, len(self.asr_mask))
      for center in val_centers:
         vsr_mask, vcr_mask = get_region_mask(higgs, center, self.sr_edge, self.cr_edge)
         self.vsr_mask = np.logical_or(self.vsr_mask, vsr_mask)
         self.vcr_mask = np.logical_or(self.vcr_mask, vcr_mask)
         vsr_masks.append(vsr_mask)
         vcr_masks.append(vcr_mask)

      # print("vsr_mask:", self.vsr_mask)
      assert ak.any(self.vsr_mask), "No validation region found. :("

      self.score_cut = float(self.config['score']['threshold'])
      self.ls_mask = self.btag_avg < self.score_cut # ls
      self.hs_mask = self.btag_avg >= self.score_cut # hs

      self.acr_ls_mask = self.acr_mask & self.ls_mask
      self.acr_hs_mask = self.acr_mask & self.hs_mask
      self.asr_ls_mask = self.asr_mask & self.ls_mask
      self.asr_hs_mask = self.asr_mask & self.hs_mask
      if not self._is_signal:
         self.blind_mask = ~self.asr_hs_mask
         self.asr_hs_mask = np.zeros_like(self.asr_mask)
      # else: self.acr_avgbtag = self.btag_avg[self.vcr_mask]

      self.vcr_ls_masks, self.vcr_hs_masks = [], []
      self.vsr_ls_masks, self.vsr_hs_masks = [], []
      for vsr_mask, vcr_mask in zip(vsr_masks, vcr_masks):
         vcr_ls_mask, vcr_hs_mask, vsr_ls_mask, vsr_hs_mask = get_hs_ls_masks(vsr_mask, vcr_mask, self.ls_mask, self.hs_mask)

         self.vcr_ls_masks.append(vcr_ls_mask)
         self.vcr_hs_masks.append(vcr_hs_mask)
         self.vsr_ls_masks.append(vsr_ls_mask)
         self.vsr_hs_masks.append(vsr_hs_mask)

      
      self.vcr_ls_mask = np.any(self.vcr_ls_masks, axis=0)
      self.vcr_hs_mask = np.any(self.vcr_hs_masks, axis=0)
      self.vsr_ls_mask = np.any(self.vsr_ls_masks, axis=0)
      self.vsr_hs_mask = np.any(self.vsr_hs_masks, axis=0)

class SixB(Tree):

   _is_signal = True

   def __init__(self, filename, config='config/bdt_params.cfg', treename='sixBtree', model_path=model_path, feyn=True):
      super().__init__(filename, treename, config, feyn=feyn)
      
      try: self.mx = int(re.search('MX_.+MY', filename).group().split('_')[1])
      except: self.mx = int(re.search('MX-.+MY', filename).group().split('-')[1].split('_')[0])
      try: self.my = int(re.search('MY.+/', filename).group().split('_')[1].split('/')[0])
      except: 
         self.my = int(re.search('MY.+/', filename).group().split('-')[1].split('/')[0].split('_')[0])
      # self.filename = re.search('NMSSM_.+/', filename).group()[:-1]
      self.sample = latexTitle(self.mx, self.my)
      self.mxmy = self.sample.replace('$','').replace('_','').replace('= ','_').replace(', ','_').replace(' GeV','')

      samp, xsec = next( ((key,value) for key,value in xsecMap.items() if key in filename),("unk",1) )
      self.xsec = xsec
      self.lumi = lumiMap[self.year][0]
      self.scale = self.lumi * xsec / self.total
      self.cutflow_scaled = (self.cutflow * self.scale).astype(int)
      if 'Official_NMSSM' in filename:
         genEventSumw = np.unique(self.get('genEventSumw', library='np')).sum()
         genWeight = self.get('genWeight', library='np') / genEventSumw
         self.genWeight = self.lumi * xsec * genWeight
         self.PUWeight = self.get('PUWeight', library='np')
         self.PUWeight_up = self.get('PUWeight_up', library='np')
         self.PUWeight_down = self.get('PUWeight_down', library='np')
         self.PUIDWeight = self.get('PUIDWeight', library='np')
         self.PUIDWeight_up = self.get('PUIDWeight_up', library='np')
         self.PUIDWeight_down = self.get('PUIDWeight_down', library='np')
         self.triggerSF = self.get('triggerScaleFactor', library='np')
         self.triggerSF_up = self.get('triggerScaleFactorUp', library='np')
         self.triggerSF_down = self.get('triggerScaleFactorDown', library='np')
         self.scale = self.genWeight*self.PUWeight*self.PUIDWeight*self.triggerSF
         self.cutflow_scaled = (self.cutflow * self.scale.sum()).astype(int)
         self.nomWeight = self.genWeight * self.PUWeight * self.PUIDWeight * self.triggerSF


      self.resolved_mask = ak.all(self.jet_signalId[:,:6] > -1, axis=1)

      # for k, v in self.tree.items():
      #    if 'gen' in k:
      #       setattr(self, k, v.array())
      
      self.jet_higgsIdx = (self.jet_signalId) // 2

      # print("Calculating SF correction factors")
      if 'Official' in self.filename: 
         self.get_sf_ratios()

   def get_sf_ratios(self):
      sf_dir = f'data/btag/MX_{self.mx}_MY_{self.my}.root'
      sf_file = uproot.open(sf_dir)

      for key in sf_file.keys():
         sf_name = key.split(f'{self.my}_')[-1].split(';')[0]
         # print(sf_name)
         
         ratio, n_bins, ht_bins = sf_file[key].to_numpy()
   
         n_jet = self.n_jet.to_numpy()
         ht = self.get('PFHT', library='np')

         i = np.digitize(n_jet, n_bins) - 1
         j = np.digitize(ht, ht_bins) - 1
         i = np.where(i == len(n_bins)-1, len(n_bins)-2, i)
         j = np.where(j == len(ht_bins)-1, len(ht_bins)-2, j)
         i = np.where(i < 0, 0, i)
         j = np.where(j < 0, 0, j)
   
         corr = ratio[i,j]
      
         raw_sf = self.tree[sf_name].array()

         setattr(self, sf_name, corr * raw_sf)
         setattr(self, f'{sf_name}_raw', raw_sf)
         self.nomWeight = self.nomWeight * self.bSFshape_central

   def sr_hist(self):
      fig, ax = plt.subplots()
      n = Hist(self.X_m[self.asr_hs_mask], bins=self.mBins, ax=ax, density=False, weights=self.scale, zorder=2)
      ax.set_xlabel(r"M$_\mathrm{X}$ [GeV]")
      ax.set_ylabel("AU")
      return fig, ax, n
      # plt.close(fig)
      # return np.where(n == 0, 1e-3, n)

   def hist(self, var, bins, label=None, ax=None, title=True, lines=None, density=False, **kwargs):
   
      if ax is None: fig, ax = plt.subplots()

      if isinstance(var, list):
         if 'colors' not in kwargs.keys(): 
            colors = [None]*len(var)
         else: colors = kwargs['colors']
         if 'styles' not in kwargs.keys(): 
            styles = [None]*len(var)
         else: styles = kwargs['styles']

         n_max = 0
         n_all = []
         for arr,lbl,c,s in zip(var,label,colors,styles):
            if isinstance(arr, str): arr = self.get(arr)
            n = Hist(arr, bins=bins, label=lbl, ax=ax, weights=self.scale, color=c, linestyle=s, density=density)
            n_all.append(n)
            if n.max() > n_max: n_max = n.max()
         ax.legend()
         return n_all
      elif isinstance(var, str):
         var = self.get(var)
      
      if label is not None:
         n = Hist(var, bins=bins, ax=ax, weights=self.scale, label=label, density=density)
      else:
         n = Hist(var, bins=bins, ax=ax, weights=self.scale, density=density)
      n_max = n.max()

      if isinstance(lines, list):
         for val in lines:
            ax.plot([val,val],[0, n_max], 'gray', '--', alpha=0.5)
      # print(np.around(n))
      
      if title:
         ax.set_title(self.sample)

      if np.issubdtype(bins.dtype, int):
         ax.tick_params(axis='x', which='minor', color='white')
         ax.xaxis.set_ticks(bins)
         # ax.xaxis.set_tick_params(which='minor', bottom=False)
      
      ax.set_ylabel('AU')

      return n

class Data(Tree):

   _is_signal = False

   def __init__(self, filename, cfg='config/bdt_params.cfg', treename='sixBtree', model_path=model_path, feyn=True):
      super().__init__(filename, treename, cfg, feyn=feyn)

      self.sample = rf"{round(lumiMap[self.year][0]/1000,1)} fb$^{{-1}}$ (13 TeV, {self.year})"
      self.set_bdt_params(cfg)
      self.set_var_dict()

   def set_bdt_params(self, cfg):
      if isinstance(cfg, str):
         config = ConfigParser()
         config.optionxform = str
         config.read(cfg)
         self.config = config
      elif isinstance(cfg, ConfigParser):
         self.config = cfg

      minMX = int(self.config['plot']['minMX'])
      maxMX = int(self.config['plot']['maxMX'])
      if self.config['plot']['style'] == 'linspace':
         nbins = int(self.config['plot']['edges'])
         self.mBins = np.linspace(minMX,maxMX,nbins)
      if self.config['plot']['style'] == 'arange':
         step = int(self.config['plot']['steps'])
         self.mBins = np.arange(minMX,maxMX,step)

      self.x_mBins = (self.mBins[:-1] + self.mBins[1:])/2

      self.Nestimators  = int(self.config['BDT']['Nestimators'])
      self.learningRate = float(self.config['BDT']['learningRate'])
      self.maxDepth     = int(self.config['BDT']['maxDepth'])
      self.minLeaves    = int(self.config['BDT']['minLeaves'])
      self.GBsubsample  = float(self.config['BDT']['GBsubsample'])
      self.randomState  = int(self.config['BDT']['randomState'])
      variables = self.config['BDT']['variables']
      if isinstance(variables, str):
         variables = variables.split(", ")
      self.variables = variables

   def set_var_dict(self):
      pt6bsum = self.HX.b1.pt + self.HX.b2.pt + self.H1.b1.pt + self.H1.b2.pt + self.H2.b1.pt + self.H2.b2.pt

      self.var_dict = {
         'pt6bsum' : self.pt6bsum,
         'dR6bmin' : self.dR6bmin,
         'dEta6bmax' : self.dEta6bmax,
         'HX_m' : self.HX.m,
         'H1_m' : self.H1.m,
         'H2_m' : self.H2.m,
         'HX_pt' : self.HX.pt,
         'H1_pt' : self.H1.pt,
         'H2_pt' : self.H2.pt,
         'HX_dr' : self.HX.dr,
         'H1_dr' : self.H1.dr,
         'H2_dr' : self.H2.dr,
         'HX_costheta' : self.HX.costheta,
         'H1_costheta' : self.H1.costheta,
         'H2_costheta' : self.H2.costheta,
         'HX_H1_dr' : self.HX.deltaR(self.H1),
         'H1_H2_dr' : self.H1.deltaR(self.H2),
         'H2_HX_dr' : self.H2.deltaR(self.HX),
         'HX_H1_dEta' : self.HX.deltaEta(self.H1),
         'H1_H2_dEta' : self.H1.deltaEta(self.H2),
         'H2_HX_dEta' : self.H2.deltaEta(self.HX),
         'HX_H1_dPhi' : self.HX.deltaPhi(self.H1),
         'H1_H2_dPhi' : self.H1.deltaPhi(self.H2),
         'H2_HX_dPhi' : self.H2.deltaPhi(self.HX),
         'Y_m' : self.Y.m,
      }

   def set_variables(self, var_list):
      self.variables = var_list

   def get_df(self, mask, variables):
      import pandas as pd
      features = {}
      for var in variables:
         # features[var] = abs(getattr(self, var)[mask])
         features[var] = abs(self.var_dict[var][mask])
      df = pd.DataFrame(features)
      return df

   def train_ar(self):
      from hep_ml import reweight
      # print(".. initializing transfer factor")
      self.AR_TF = sum(self.acr_hs_mask)/sum(self.acr_ls_mask)
      ls_weights = np.ones(ak.sum(self.acr_ls_mask))*self.AR_TF
      hs_weights = np.ones(ak.sum([self.acr_hs_mask]))
      # np.digitize()  

      # print(".. initializing dataframes of variables")
      AR_df_ls = self.get_df(self.acr_ls_mask, self.variables)
      AR_df_hs = self.get_df(self.acr_hs_mask, self.variables)

      np.random.seed(self.randomState) #Fix any random seed using numpy arrays
      # print(".. calling reweight.GBReweighter")
      reweighter_base = reweight.GBReweighter(
         n_estimators=self.Nestimators, 
         learning_rate=self.learningRate, 
         max_depth=self.maxDepth, 
         min_samples_leaf=self.minLeaves,
         gb_args={'subsample': self.GBsubsample})

      # print(".. calling reweight.FoldingReweighter")
      reweighter = reweight.FoldingReweighter(reweighter_base, random_state=self.randomState, n_folds=2, verbose=False)

      # print(".. calling reweighter.fit for AR")
      reweighter.fit(AR_df_ls,AR_df_hs,ls_weights,hs_weights)
      self.reweighter = reweighter
      # self.AR_reweighter = reweighter

      # print(".. predicting AR hs weights")
      AR_df_ls = self.get_df(self.asr_ls_mask, self.variables)
      initial_weights = np.ones(ak.sum(self.asr_ls_mask))*self.AR_TF

      self.asr_weights = reweighter.predict_weights(AR_df_ls,initial_weights,lambda x: np.mean(x, axis=0))

      self.acr_df_ls = self.get_df(self.acr_ls_mask, self.variables)
      self.acr_initial_weights = np.ones(ak.sum(self.acr_ls_mask))*self.AR_TF
      self.acr_weights = reweighter.predict_weights(self.acr_df_ls,self.acr_initial_weights,lambda x: np.mean(x, axis=0))

   def train_vr(self):
      from hep_ml import reweight
      # print(".. initializing transfer factor")
      self.vr_tf = sum(self.vcr_hs_mask)/sum(self.vcr_ls_mask)
      ls_weights = np.ones(ak.sum(self.vcr_ls_mask))*self.vr_tf
      hs_weights = np.ones(ak.sum([self.vcr_hs_mask]))

      # print(".. initializing dataframes of variables")
      df_ls = self.get_df(self.vcr_ls_mask, self.variables)
      df_hs = self.get_df(self.vcr_hs_mask, self.variables)

      np.random.seed(self.randomState) #Fix any random seed using numpy arrays
      # print(".. calling reweight.GBReweighter")
      reweighter_base = reweight.GBReweighter(
         n_estimators=self.Nestimators, 
         learning_rate=self.learningRate, 
         max_depth=self.maxDepth, 
         min_samples_leaf=self.minLeaves,
         gb_args={'subsample': self.GBsubsample})

      # print(".. calling reweight.FoldingReweighter")
      reweighter = reweight.FoldingReweighter(reweighter_base, random_state=self.randomState, n_folds=2, verbose=False)

      # print(".. calling reweighter.fit")
      reweighter.fit(df_ls,df_hs,ls_weights,hs_weights)
      # self.VR_reweighter = reweighter

      # print(".. predicting VR hs weights")
      vcr_df_ls = self.get_df(self.vcr_ls_mask, self.variables)
      vcr_initial_weights = np.ones(ak.sum(self.vcr_ls_mask))*self.vr_tf
      self.vcr_weights = reweighter.predict_weights(vcr_df_ls,vcr_initial_weights,lambda x: np.mean(x, axis=0))

      self.vsr_df_ls = self.get_df(self.vsr_ls_mask, self.variables)
      self.vsr_initial_weights = np.ones(ak.sum(self.vsr_ls_mask))*self.vr_tf
      self.vsr_weights = reweighter.predict_weights(self.vsr_df_ls,self.vsr_initial_weights,lambda x: np.mean(x, axis=0))

   def train(self):
      self.set_var_dict()
      print(f"high avg b tag score threshold = {self.score_cut}")
      print(".. training in validation region")
      self.train_vr()
      print(".. training in analysis region")
      self.train_ar()

      self.vr_stat_prec = round(1 + np.sqrt(abs(1/self.vsr_weights.sum() - 1/self.asr_weights.sum())), 2)
      err = np.sqrt(np.sum(self.vsr_weights**2))
      self.total_error = np.sqrt(self.vsr_weights.sum() + err**2 + (self.vsr_weights.sum()*self.vrtf)**2)
      self.diff = self.vsr_hs_mask.sum() - self.vsr_weights.sum()
      sd = self.diff / self.total_error
      if self.diff**2 > self.total_error: self.K = np.sqrt(self.diff**2 - self.total_error**2)
      else: self.K = 0
      self.vr_yield_val = round(1 + self.K / self.asr_weights.sum(), 2)

      self.vsr_model_ks_t, self.vsr_model_ks_p = self.ks_test('X_m', self.vsr_ls_mask, self.vsr_hs_mask, self.vsr_weights) # weighted ks test (ls shape with weights)
      self.vsr_const_ks_t, self.vsr_const_ks_p = self.ks_test('X_m', self.vsr_ls_mask, self.vsr_hs_mask, np.ones_like(self.vsr_weights)) # unweighted ks test (ls shape without weights)
      self.vcr_model_ks_t, self.vcr_model_ks_p = self.ks_test('X_m', self.vcr_ls_mask, self.vcr_hs_mask, self.vcr_weights)

   def vr_histo(self, ls_mask, hs_mask, weights, density=False, vcr=False, data_norm=False, ax=None, variable='X_m', bins=None, norm=None):
      # ratio = ak.sum(self.vcr_ls_mask) / ak.sum(self.vcr_hs_mask) 
      ratio = 1
      # fig, axs = plt.subplots()

      # if ax is None: 
      fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})

      # fig.suptitle("Validation Control Region")
      var = getattr(self, variable)
      var = self.X.m
      xlabel = xlabel_dict[variable]
      if variable == 'X_m': bins = self.mBins
      else: bins = bin_dict[variable]
      bins=bins
      
      n_ls = np.histogram(var[ls_mask], bins=bins)[0]
      n_hs = np.histogram(var[hs_mask], bins=bins)[0]
      # print(n_ls.sum(), n_hs.sum(), weights.sum())
      # print(np.histogram(var[ls_mask], bins=bins, weights=weights)[0])
      # print(weights, weights.shape)
      n_target, n_model, n_ratio, total = Ratio([var[ls_mask], var[hs_mask]], weights=[weights*ratio, None], bins=bins, density=density, axs=axs, labels=['Model', 'Data'], xlabel=r"M$_\mathrm{X}$ [GeV]", ratio_ylabel='Obs/Exp', data_norm=data_norm, norm=norm, total=True)

      axs[0].set_ylabel('Events')

      axs[1].set_xlabel(xlabel)

      if vcr: 
         self.bin_ratios = np.nan_to_num(n_hs / n_ls)
         # print("bin ratios",self.bin_ratios)
      # else: 
      #    print(n_ls.sum(), n_hs.sum(), weights.sum(), (n_ls*self.bin_ratios).sum())


      sumw2 = []
      err = []
      for i,n_nominal in enumerate(n_model):#, model_uncertainty_up, model_uncertainty_down)):
         low_x = self.X_m[self.vsr_ls_mask] > self.mBins[i]
         high_x = self.X_m[self.vsr_ls_mask] <= self.mBins[i+1]
         weights = np.sum((self.vsr_weights[low_x & high_x]/total)**2)
         sumw2.append(weights)
         weights = np.sqrt(weights)
         # print(weights, n_nominal)
         err.append(weights)
         model_uncertainty_up = n_nominal + weights
         model_uncertainty_down = n_nominal - weights
         # print(model_uncertainty_up,model_uncertainty_down)

         # axs[0].fill_between([self.mBins[i], self.mBins[i+1]], model_uncertainty_down, model_uncertainty_up, color='C0', alpha=0.25)
         ratio_up   = np.nan_to_num(model_uncertainty_up / n_nominal)
         # print(weights, model_uncertainty_up, n_nominal)
         ratio_down = np.nan_to_num(model_uncertainty_down / n_nominal)
         # print(ratio_up, ratio_down)
         axs[1].fill_between([self.mBins[i], self.mBins[i+1]], ratio_down, ratio_up, color='C0', alpha=0.25)

      self.VR_sumw2 = np.array((sumw2))
      self.VR_err = np.array((err))

      return fig, axs, n_target, n_model, n_ratio

   def ks_test(self, variable, ls_mask, hs_mask, weights):
      from scipy.stats import kstwobign
      from awkward.highlevel import Array

      ratio = 1

      var = getattr(self, variable)

      sort_mask_ls = np.argsort(var[ls_mask])
      sort_mask_hs = np.argsort(var[hs_mask])

      sorted_ls_var = var[ls_mask][sort_mask_ls]
      sorted_hs_var = var[hs_mask][sort_mask_hs]

      if isinstance(sorted_ls_var, Array): sorted_ls_var = sorted_ls_var.to_numpy()
      if isinstance(sorted_hs_var, Array): sorted_hs_var = sorted_hs_var.to_numpy()

      sorted_weights = weights[sort_mask_ls]*ratio
      x_ls = sorted_ls_var
      y_model = sorted_weights.cumsum() / sorted_weights.sum()

      x_hs = sorted_hs_var
      y_obs = np.ones_like(x_hs).cumsum() / hs_mask.sum()

      ess_unweighted = hs_mask.sum()
      ess_weighted = np.sum(sorted_weights)**2 / np.sum(weights**2)

      nm = np.sqrt((ess_weighted*ess_unweighted)/(ess_weighted + ess_unweighted))

      x_max = max(x_ls.max(), x_hs.max())
      x_min = min(x_ls.min(), x_hs.min())
      x_bins = np.linspace(x_min, x_max, 100)
      y_model_interp = np.interp(x_bins, x_ls, y_model)
      y_obs_interp = np.interp(x_bins, x_hs, y_obs)

      y_model_interp = y_model_interp
      y_obs_interp = y_obs_interp

      difference = abs(y_model_interp - y_obs_interp) * nm

      ks_statistic = round(difference.max(),2)
      try: ks_probability = int(kstwobign.sf(ks_statistic)*100)
      except: ks_probability = 0

      return ks_statistic, ks_probability
      
   def vsr_hist(self):
      fig, axs, n_target, n_model, n_ratio = self.vr_hist(self.vsr_ls_mask, self.vsr_hs_mask, self.vsr_weights, density=False)
      axs[0].set_title('Validation Signal Region')
      return fig, axs, n_target, n_model
   
   def vcr_hist(self):
      fig, axs, n_target, n_model, n_ratio = self.vr_hist(self.vcr_ls_mask, self.vcr_hs_mask, self.vcr_weights, density=False, vcr=True)
      axs[0].set_title('Validation Control Region')
      return fig, axs, n_target, n_model

   def pull_plots(self, variable='X_m', saveas=None, region='VSR'):
      from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
      fig_before = plt.figure(figsize=(30,10))
      fig_after = plt.figure(figsize=(30,10))

      GS_before = GridSpec(1, 3, figure=fig_before, width_ratios=[4,5,5])
      GS_after = GridSpec(1, 3, figure=fig_after, width_ratios=[4,5,5])

      gs1_b = GridSpecFromSubplotSpec(2, 1, subplot_spec=GS_before[0], height_ratios=[3,1])
      gs2_b = GridSpecFromSubplotSpec(1, 1, subplot_spec=GS_before[1])
      gs3_b = GridSpecFromSubplotSpec(1, 1, subplot_spec=GS_before[2])
      gs1_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=GS_after[0], height_ratios=[3,1])
      gs2_a = GridSpecFromSubplotSpec(1, 1, subplot_spec=GS_after[1])
      gs3_a = GridSpecFromSubplotSpec(1, 1, subplot_spec=GS_after[2])

      ax1t_b = fig_before.add_subplot(gs1_b[0])
      ax1b_b = fig_before.add_subplot(gs1_b[1])
      ax2_b = fig_before.add_subplot(gs2_b[0])
      ax3_b = fig_before.add_subplot(gs3_b[0])
      
      ax1t_a = fig_after.add_subplot(gs1_a[0])
      ax1b_a = fig_after.add_subplot(gs1_a[1])
      ax2_a = fig_after.add_subplot(gs2_a[0])
      ax3_a = fig_after.add_subplot(gs3_a[0])

      # fig_before, axs_before = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})
      # fig_after, axs_after = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})

      ax1t_a.set_title(self.sample)
      ax1t_b.set_title(self.sample)

      # var = getattr(self, variable)
      # var = self.var_dict[variable]
      var = self.X.m

      # xlabel = xlabel_dict[variable]
      if variable == 'X_m': bins = self.mBins
      else: bins = bin_dict[variable]
      bins=bins

      region_dict = {
         'VSR' : {
            'title' : 'Validation Signal Region',
            'ls_mask' : self.vsr_ls_mask,
            'hs_mask' : self.vsr_hs_mask,
            'weights' : self.vsr_weights,
            'crhs_mask' : self.vcr_hs_mask,
            'crls_mask' : self.vcr_ls_mask
         },
         'VCR' : {
            'title' : 'Validation Control Region',
            'ls_mask' : self.vcr_ls_mask,
            'hs_mask' : self.vcr_hs_mask,
            'weights' : self.vcr_weights,
            'crhs_mask' : self.vcr_hs_mask,
            'crls_mask' : self.vcr_ls_mask
         },
         'ACR' : {
            'title' : 'Analysis Control Region',
            'ls_mask' : self.acr_ls_mask,
            'hs_mask' : self.acr_hs_mask,
            'weights' : self.acr_weights,
            'crhs_mask' : self.acr_hs_mask,
            'crls_mask' : self.acr_ls_mask
         },
      }

      ls_mask = region_dict[region]['ls_mask']
      hs_mask = region_dict[region]['hs_mask']
      weights = region_dict[region]['weights']
      crhs_mask = region_dict[region]['crhs_mask']
      crls_mask = region_dict[region]['crls_mask']
      title = region_dict[region]['title']

      ax2_b.set_title(f'{title} - Before BDT Weights')
      ax3_b.set_title(f'{title} - Before BDT Weights')
      ax2_a.set_title(f'{title} - After BDT Weights')
      ax3_a.set_title(f'{title} - After BDT Weights')

      n_ratio = np.ones_like(var[ls_mask]) * sum(crhs_mask) / sum(crls_mask)

      from utils.plotter import model_ratio, plot_residuals, plot_pulls

      h_pred, h_target, ratio_b, err_pred = model_ratio(var[hs_mask], var[ls_mask], n_ratio, bins=bins, ax_top=ax1t_b, ax_bottom=ax1b_b)
      n_model, n_target, ratio_a, err_model = model_ratio(var[hs_mask], var[ls_mask], weights, bins=bins, ax_top=ax1t_a, ax_bottom=ax1b_a)

      self.norm_err = 1 + np.average(1-ratio_a)

      # Ratio Plot (Middle axis)
      plot_residuals(ratio_b, ax2_b)
      self.norm_err = plot_residuals(ratio_a, ax2_a)

      # Pull plot (Right Plot)
      plot_pulls(h_pred, h_target, ax3_b, err_pred)
      plot_pulls(n_model, n_target, ax3_a, err_model)

      plt.tight_layout()

      if saveas is not None:
         # fname = f"{savein}/{filename}_{variable}_before.pdf"
         # fig_before.savefig(saveas, bbox_inches='tight')
         # fname = f"{savein}/{filename}_{variable}_after.pdf"
         fig_after.savefig(saveas, bbox_inches='tight')


   def before_after(self, savedir=None, variable='X_m'):

      # xlabel = xlabel_dict[variable]

      # fig, axs, n_obs, n_unweighted
      fig, axs, n_target, n_model, n_ratio = self.vr_hist(self.vsr_ls_mask, self.vsr_hs_mask, np.ones_like(self.vsr_weights)/sum(self.vsr_ls_mask), data_norm=True, variable=variable)
      axs[0].set_title('Validation Signal Region - Before Applying BDT Weights', fontsize=18)

      if savedir is not None: fig.savefig(f"{savedir}/{variable}_vsr_before_bdt.pdf")

      fig, axs, n_target, n_model, n_ratio = self.vr_hist(self.vsr_ls_mask, self.vsr_hs_mask, self.vsr_weights/self.vsr_weights.sum(), data_norm=True, variable=variable)
      axs[0].set_title('Validation Signal Region - After Applying BDT Weights', fontsize=18)
      if savedir is not None: fig.savefig(f"{savedir}/{variable}_vsr_after_bdt.pdf")

      # fig.savefig(f"plots/model_VCR.pdf", bbox_inches='tight')

   def sr_hist(self, savein=None):
      from matplotlib.lines import Line2D
      from matplotlib.patches import Patch

      fig, axs = plt.subplots(nrows=2,  gridspec_kw={'height_ratios':[4,1]})

      n_model_SR_hs = Hist(self.X_m[self.asr_ls_mask], weights=self.asr_weights, bins=self.mBins, ax=axs[0], label='asr', density=False, color='C2')
      weights2 = np.histogram(self.X_m[self.asr_ls_mask], weights=self.asr_weights**2, bins=self.mBins)[0]
      self.error = np.sqrt(weights2)

      model_uncertainty = np.sqrt(n_model_SR_hs)

      sumw2 = []
      err = []
      for i,n_nominal in enumerate(n_model_SR_hs):#, model_uncertainty_up, model_uncertainty_down)):
         low_x = self.X_m[self.asr_ls_mask] > self.mBins[i]
         high_x = self.X_m[self.asr_ls_mask] <= self.mBins[i+1]
         weights = np.sum(self.asr_weights[low_x & high_x]**2)
         sumw2.append(weights)
         weights = np.sqrt(weights)
         err.append(weights)
         model_uncertainty_up = n_nominal + weights
         model_uncertainty_down = n_nominal - weights

         axs[0].fill_between([self.mBins[i], self.mBins[i+1]], model_uncertainty_down, model_uncertainty_up, color='C2', alpha=0.25)
         
         ratio_up = np.nan_to_num(model_uncertainty_up / n_nominal)
         ratio_down = np.nan_to_num(model_uncertainty_down / n_nominal)
         # print(ratio_down)
         # print(i, i+1, ratio_down, ratio_up)
         axs[1].fill_between([self.mBins[i], self.mBins[i+1]], ratio_down, ratio_up, color='C2', alpha=0.25)

      self.sumw2 = np.array((sumw2))
      self.err = np.array((err))

      model_nominal = Line2D([0], [0], color='C2', lw=2, label='Bkg Model')
      handles = [model_nominal, Patch(facecolor='C2', alpha=0.25, label='Bkg Uncertainty')]
      
      axs[0].legend(handles=handles)

      axs[0].set_ylabel('Events')
      axs[1].set_ylabel('Uncertainty')

      axs[1].plot([self.mBins[0], self.mBins[-1]], [1,1], color='gray', linestyle='--')
      axs[1].set_xlabel(r"$M_X$ [GeV]")

      if savein is not None: ROOTHist(n_model_SR_hs, 'data', savein)
      
      return fig, axs, n_model_SR_hs

   def vr_hist(self, savein=None):
      from matplotlib.lines import Line2D
      from matplotlib.patches import Patch

      fig, axs = plt.subplots(nrows=2,  gridspec_kw={'height_ratios':[4,1]})

      x = (self.mBins[1:] + self.mBins[:-1])/2
      n_model_SR_hs = Hist(self.X_m[self.vsr_ls_mask], weights=self.vsr_weights, bins=self.mBins, ax=axs[0], label='Model vsr', density=False)
      # n_SR_hs = Hist(self.X_m[self.vsr_hs_mask], bins=self.mBins, ax=axs[0], label='Target vsr', density=False, color='k')
      n_SR_hs, _ = np.histogram(self.X_m[self.vsr_hs_mask], bins=self.mBins)
      print(len(x),len(n_SR_hs))
      axs[0].scatter(x, n_SR_hs, color='k', label='Target vsr')
      n_err = np.sqrt(n_SR_hs)
      for xval,val,nval in zip(x,n_err,n_SR_hs):
         axs[0].plot([xval,xval], [nval-val, nval+val], color='k')

      model_uncertainty = np.sqrt(n_model_SR_hs)

      sumw2 = []
      err = []
      for i,n_nominal in enumerate(n_model_SR_hs):#, model_uncertainty_up, model_uncertainty_down)):
         low_x = self.X_m[self.vsr_ls_mask] > self.mBins[i]
         high_x = self.X_m[self.vsr_ls_mask] <= self.mBins[i+1]
         weights = np.sum(self.vsr_weights[low_x & high_x]**2)
         sumw2.append(weights)
         weights = np.sqrt(weights)
         err.append(weights)
         model_uncertainty_up = n_nominal + weights
         model_uncertainty_down = n_nominal - weights

         axs[0].fill_between([self.mBins[i], self.mBins[i+1]], model_uncertainty_down, model_uncertainty_up, color='C0', alpha=0.25)
         
         ratio_up = np.nan_to_num(model_uncertainty_up / n_nominal)
         ratio_down = np.nan_to_num(model_uncertainty_down / n_nominal)
         # print(ratio_down)
         # print(i, i+1, ratio_down, ratio_up)
         axs[1].fill_between([self.mBins[i], self.mBins[i+1]], ratio_down, ratio_up, color='C0', alpha=0.25)

      self.sumw2 = np.array((sumw2))
      self.err = np.array((err))

      model_nominal = Line2D([0], [0], color='C0', lw=2, label='Validation Bkg Model')
      target_handle = Line2D([0], [0], marker='o', color='w', label='Validation Target Distribution', markerfacecolor='k', markersize=8)

      handles = [target_handle, model_nominal, Patch(facecolor='C0', alpha=0.25, label='Validation Bkg Uncertainty')]
      
      axs[0].legend(handles=handles)

      axs[0].set_ylabel('Events')
      axs[1].set_ylabel('Uncertainty')

      axs[1].plot([self.mBins[0], self.mBins[-1]], [1,1], color='gray', linestyle='--')
      axs[1].set_xlabel(r"$M_X$ [GeV]")

      if savein is not None: ROOTHist(n_model_SR_hs, 'data', savein)
      
      return fig, axs, n_model_SR_hs
      


bins = 31

bin_dict = {
   'pt6bsum' : np.linspace(300,1000,bins),
   'dR6bmin' : np.linspace(0,2,bins),
   'dEta6bmax' : np.linspace(0,4,bins),
   'HX_pt' : np.linspace(0,400,bins),
   'H1_pt' : np.linspace(0,300,bins),
   'H2_pt' : np.linspace(0,300,bins),
   'HX_m' : np.linspace(125,250,bins),
   'H1_m' : np.linspace(125,250,bins),
   'H2_m' : np.linspace(125,250,bins),
   'H2_pt' : np.linspace(0,300,bins),
   'H1_dr' : np.linspace(0,4,bins),
   'HX_dr' : np.linspace(0,4,bins),
   'H2_dr' : np.linspace(0,4,bins),
   'HX_H1_dEta' : np.linspace(0, 5, bins),
   'H1_H2_dEta' : np.linspace(0, 5, bins),
   'H2_HX_dEta' : np.linspace(0, 5, bins),
   'HX_H1_dPhi' : np.linspace(-np.pi, np.pi, bins),
   'HX_costheta' : np.linspace(-1,1,bins),
   'H1_costheta' : np.linspace(-1,1,bins),
   'H2_costheta' : np.linspace(-1,1,bins),
   'Y_m' : np.linspace(250, 900, bins),
   'X_m' : np.linspace(375, 2000, bins),
}

xlabel_dict = {
   'X_m' : r"$M_X$ [GeV]",
   'pt6bsum' : r"$\Sigma_i^{jets} p_{T,i}$ [GeV]",
   'dR6bmin' : r"$min(\Delta R_{jj})$",
   'HX_pt' : r"$H_X \; p_T$ [GeV]",
   'H1_pt' : r"$H_1 \; p_T$ [GeV]",
   'H2_pt' : r"$H_2 \; p_T$ [GeV]",
   'H1_m' : r"$H_1$ mass [GeV]",
   'HX_m' : r"$H_X$ mass [GeV]",
   'HX_dr' : r"$H_X \Delta R_{bb}$",
   'H2_dr' : r"$H_1 \Delta R_{bb}$",
   'H2_dr' : r"$H_2 \Delta R_{bb}$",
   'H1_H2_dEta' : r"$\Delta\eta(H_1,H_2)$",
   'Y_m' : r"$M_Y$ [GeV]"
}

