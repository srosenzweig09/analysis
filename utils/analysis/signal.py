"""
Author: Suzanne Rosenzweig
"""

model_version = 'old'
# model_version = 'new'

import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from utils.useCMSstyle import *
plt.style.use(CMS)
from utils.plotter import Hist
from utils.analysis.particle import Particle, Higgs, Y
from utils.analysis.feyn import Model
from utils.xsecUtils import lumiMap, xsecMap
from configparser import ConfigParser
from utils.analysis.bdt import BDTRegion

import re
import os, sys 
from colorama import Fore, Style
# import subprocess

def latexTitle(mx,my):
   #  ind = -2
   #  if 'output' in descriptor: ind = -3
   #  mass_point = descriptor.split("/")[ind]
   #  mX = mass_point.split('_')[ind-1]
   #  mY = mass_point.split('_')[ind+1]
    return r"$M_X=$ " + str(mx) + r" GeV, $M_Y=$ " + str(my) + " GeV"

def get_indices(var, bins):
   ind = np.digitize(var, bins) - 1
   ind = np.where(ind == len(bins)-1, len(bins)-2, ind)
   ind = np.where(ind < 0, 0, ind)
   return ind

def ROOTHist(h_vals, title, filepath):
   """
   title : should be either 'signal' or 'data'
   """
   import ROOT
   ROOT.gROOT.SetBatch(True)
   from array import array
   assert ".root" in filepath, print("[ERROR] Please include '.root' in filepath")

   nbins = 40
   m_bins = np.linspace(375, 1150, nbins)
   x_X_m = (m_bins[:-1] + m_bins[1:]) / 2
   
   fout = ROOT.TFile(f"{filepath}","recreate")
   fout.cd()

   canvas = ROOT.TCanvas('c1','c1', 600, 600)
   canvas.SetFrameLineWidth(3)
   canvas.Draw()

   h_title = title
   ROOT_hist = ROOT.TH2D(h_title,";m_{X} [GeV];Events",nbins-1,array('d',list(m_bins)))
   for i,(val) in enumerate(h_vals):
      ROOT_hist.SetBinContent(i+1, val) 

   ROOT_hist.Draw("hist")
   ROOT_hist.Write()
   fout.Close()

# Parent Class
class Tree():
   _is_signal = False
   _is_bkg = False
   _is_data = False

   def __init__(self, filepath, treename='sixBtree', cfg=None, feyn=True):
      from utils.cutConfig import btagWP
      self.btagWP = btagWP

      self.filepath = filepath
      print(f"{Fore.CYAN}ntuple: {self.filepath}{Style.RESET_ALL}")
      self.filename = self.filepath.split('/')[-2]
      self.treename = treename
      self.year = int([yr for yr in ['2016', '2017', '2018'] if yr in filepath][0])
      self.year_long = re.search("ntuples/(.*?)/", filepath).group(1)
      self.tree = uproot.open(f"{filepath}:{treename}")

      if cfg is None: cfg = f'config/bdt_{self.year_long}.cfg'
      self.read_config(cfg)

      pattern = re.compile('H.+_') # Search for any keys beginning with an 'H' and followed somewhere by a '_'
      for k, v in self.tree.items():
         if re.match(pattern, k) or 'jet' in k or 'gen' in k or 'Y' in k or 'X' in k:
            setattr(self, k, v.array())
      with uproot.open(f"{filepath}:h_cutflow_unweighted") as f: cutflow = f
      with uproot.open(f"{filepath}:h_cutflow") as f: cutflow_w = f


      self.cutflow_w = cutflow_w
      self.cutflow_labels = cutflow.axis().labels()

      self.nevents = int(cutflow.to_numpy()[0][-1])
      if len(self.tree['jet_pt'].array()) == 0: self.nevents = 0

      self.total = int(cutflow.to_numpy()[0][0])

      self.cutflow = (cutflow.to_numpy()[0]).astype(int)
      self.cutflow_norm = (cutflow.to_numpy()[0]/cutflow.to_numpy()[0][0]*100).astype(int)

      key = str(self.year)
      if key == '2016' and 'preVFP' in self.filepath: key += 'preVFP'
      elif key == '2016': key += 'postVFP'
      self.year_key = key
      self.loose_wp = self.btagWP[key]['Loose']
      self.medium_wp = self.btagWP[key]['Medium']
      self.tight_wp = self.btagWP[key]['Tight']

      btag_mask = ak.argsort(self.jet_btag, axis=1, ascending=False) < 6
      btag = self.jet_btag[btag_mask]
      self.btag_avg = ak.mean(btag, axis=1)

      if self.nevents > 0:
      # if feyn: self.model = Model(model_version, self) # particles initialized here
         if isinstance(feyn, bool) and feyn: self.model = Model(tree=self) # particles initialized here
         elif isinstance(feyn, str): self.model = Model(tree=self, config=feyn) # particles initialized here
         else: self.initialize_bosons()
         self.bdt = BDTRegion(self)
      
      if (self.nevents != 0) & (self.nevents != len(self.n_jet)): print("WARNING! Number of events in cutflow does not match number of events in tree!")

   def read_config(self, cfg):
      if isinstance(cfg, str):
         config = ConfigParser()
         config.optionxform = str
         config.read(cfg)
         self.config = config
      elif isinstance(cfg, ConfigParser):
         self.config = cfg

   def keys(self):
      return uproot.open(f"{self.filepath}:{self.treename}").keys()

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

   def init_unregressed(self):
      btag_mask = ak.argsort(self.jet_btag, axis=1, ascending=False) < 6

      pt = self.jet_pt[btag_mask][self.combos]
      phi = self.jet_phi[btag_mask][self.combos]
      eta = self.jet_eta[btag_mask][self.combos]
      m = self.jet_m[btag_mask][self.combos]
      btag = self.jet_btag[btag_mask][self.combos]
      sig_id = self.jet_signalId[btag_mask][self.combos]
      h_id = (self.jet_signalId[btag_mask][self.combos] + 2) // 2

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

      H1_b1 = {'pt':sample_particles[0].pt,'eta':sample_particles[0].eta,'phi':sample_particles[0].phi,'m':sample_particles[0].m,'btag':sample_particles[0].btag,'sig_id':sample_particles[0].sig_id,'h_id':sample_particles[0].h_id}
      H1_b2 = {'pt':sample_particles[1].pt,'eta':sample_particles[1].eta,'phi':sample_particles[1].phi,'m':sample_particles[1].m,'btag':sample_particles[1].btag,'sig_id':sample_particles[1].sig_id,'h_id':sample_particles[1].h_id}
      H2_b1 = {'pt':sample_particles[2].pt,'eta':sample_particles[2].eta,'phi':sample_particles[2].phi,'m':sample_particles[2].m,'btag':sample_particles[2].btag,'sig_id':sample_particles[2].sig_id,'h_id':sample_particles[2].h_id}
      H2_b2 = {'pt':sample_particles[3].pt,'eta':sample_particles[3].eta,'phi':sample_particles[3].phi,'m':sample_particles[3].m,'btag':sample_particles[3].btag,'sig_id':sample_particles[3].sig_id,'h_id':sample_particles[3].h_id}
      H3_b1 = {'pt':sample_particles[4].pt,'eta':sample_particles[4].eta,'phi':sample_particles[4].phi,'m':sample_particles[4].m,'btag':sample_particles[4].btag,'sig_id':sample_particles[4].sig_id,'h_id':sample_particles[4].h_id}
      H3_b2 = {'pt':sample_particles[5].pt,'eta':sample_particles[5].eta,'phi':sample_particles[5].phi,'m':sample_particles[5].m,'btag':sample_particles[5].btag,'sig_id':sample_particles[5].sig_id,'h_id':sample_particles[5].h_id}

      self.H1_unregressed = Higgs(H1_b1, H1_b2)

      # self.H1_b1 = self.H1.b1
      # self.H1_b2 = self.H1.b2

      H2 = Higgs(H2_b1, H2_b2)
      H3 = Higgs(H3_b1, H3_b2)

      assert ak.all(self.H1.b1.pt >= self.H1.b2.pt)
      assert ak.all(H2.b1.pt >= H2.b2.pt)
      assert ak.all(H3.b1.pt >= H3.b2.pt)

      self.Y_unregressed = Y(H2, H3)

      self.H2_unregressed = self.Y_unregressed.H2
      self.H3_unregressed = self.Y_unregressed.H3

      assert ak.all(self.H2.pt >= self.H3.pt)

   def get_regressed_kinematics(self, particle_name):
      pt = self.tree[f'{particle_name}_pt'].array()
      eta = self.tree[f'{particle_name}_eta'].array()
      phi = self.tree[f'{particle_name}_phi'].array()
      m = self.tree[f'{particle_name}_m'].array()
      btag = self.tree[f'{particle_name}_btag'].array()

      ind = (self.jet_pt == pt) & (self.jet_eta == eta) & (self.jet_phi == phi) & (self.jet_m == m)
      assert ak.all(ak.sum(ind, axis=1) == 1)
      
      ptRegressed = ak.flatten(self.jet_ptRegressed[ind])
      mRegressed = ak.flatten(self.jet_mRegressed[ind])
      sig_id = ak.flatten(self.jet_signalId[ind])
      h_id = (sig_id + 2) // 2
      
      return {'pt' : ptRegressed, 'eta' : eta, 'phi' : phi, 'm' : mRegressed, 'btag' : btag, 'sig_id' : sig_id, 'h_id' : h_id}

   def initialize_bosons(self):

      self.H1 = Higgs(self.get_regressed_kinematics('HX_b1'), self.get_regressed_kinematics('HX_b2'))
      self.H2 = Higgs(self.get_regressed_kinematics('H1_b1'), self.get_regressed_kinematics('H1_b2'))
      self.H3 = Higgs(self.get_regressed_kinematics('H2_b1'), self.get_regressed_kinematics('H2_b2'))

      h1_found = (self.H1.b1.h_id == self.H1.b2.h_id) & (self.H1.b1.h_id > 0)
      h2_found = (self.H2.b1.h_id == self.H2.b2.h_id) & (self.H2.b1.h_id > 0)
      h3_found = (self.H3.b1.h_id == self.H3.b2.h_id) & (self.H3.b1.h_id > 0)
      self.n_h_found = h1_found*1 + h2_found*1 + h3_found*1

      self.Y = Particle(self, 'Y')

      self.X = Particle(self, 'X')
      
      self.h_id = (self.jet_signalId + 2) // 2
      hx_possible = ak.sum(self.h_id == 1, axis=1) == 2
      h1_possible = ak.sum(self.h_id == 2, axis=1) == 2
      h2_possible = ak.sum(self.h_id == 3, axis=1) == 2
      self.n_h_possible = hx_possible*1 + h1_possible*1 + h2_possible*1

      self.H1_dr = self.H1.dr
      self.H2_dr = self.H2.dr
      self.H3_dr = self.H3.dr

      self.H1_m = self.H1.m
      self.H2_m = self.H2.m
      self.H3_m = self.H3.m

      self.H1_H2_dEta = self.H1.deltaEta(self.H2)
      self.H2_H3_dEta = self.H2.deltaEta(self.H3)
      self.H3_H1_dEta = self.H3.deltaEta(self.H1)

      self.H1_H2_dPhi = self.H1.deltaPhi(self.H2)
      self.H2_H3_dPhi = self.H2.deltaPhi(self.H3)
      self.H3_H1_dPhi = self.H3.deltaPhi(self.H1)

      self.H1_costheta = self.H1.costheta
      self.H2_costheta = self.H2.costheta
      self.H3_costheta = self.H3.costheta

   def initialize_gen(self):
      """Initializes gen-matched particles."""

      self.gen_matched_H1_b1 = Particle(self, 'gen_H1_b1')
      self.gen_matched_H1_b2 = Particle(self, 'gen_H1_b2')
      self.gen_matched_H2_b1 = Particle(self, 'gen_H2_b1')
      self.gen_matched_H2_b2 = Particle(self, 'gen_H2_b2')
      self.gen_matched_H3_b1 = Particle(self, 'gen_H3_b1')
      self.gen_matched_H3_b2 = Particle(self, 'gen_H3_b2')

      self.gen_matched_H1 = self.gen_matched_H1_b1 + self.gen_matched_H1_b2
      self.gen_matched_H2 = self.gen_matched_H2_b1 + self.gen_matched_H2_b2
      self.gen_matched_H3 = self.gen_matched_H3_b1 + self.gen_matched_H3_b2

      good_mask = ak.nan_to_num(self.gen_matched_H1.m, nan=-1) > -1
      good_mask = good_mask & (ak.nan_to_num(self.gen_matched_H2.m, nan=-1) > -1)
      self.good_mask = good_mask & (ak.nan_to_num(self.gen_matched_H3.m, nan=-1) > -1)

      # self.resolved_mask = ak.all(self.jet_signalId[:,:6] > -1, axis=1)

class SixB(Tree):

   _is_signal = True

   def __init__(self, filepath, config=None, treename='sixBtree', feyn=True):
      super().__init__(filepath, treename, config, feyn=feyn)

      if feyn: print(f"Using FeynNet: {self.model_name}")
      else: print("[WARNING] Not using FeynNet for reconstructions!")
      
      try: self.mx = int(re.search('MX_.+MY', filepath).group().split('_')[1])
      except: self.mx = int(re.search('MX-.+MY', filepath).group().split('-')[1].split('_')[0])
      try: self.my = int(re.search('MY.+/', filepath).group().split('_')[1].split('/')[0])
      except: 
         self.my = int(re.search('MY.+/', filepath).group().split('-')[1].split('/')[0].split('_')[0])
      # self.filepath = re.search('NMSSM_.+/', filepath).group()[:-1]
      self.sample = latexTitle(self.mx, self.my)
      self.mxmy = self.sample.replace('$','').replace('_','').replace('= ','_').replace(', ','_').replace(' GeV','')

      samp, xsec = next( ((key,value) for key,value in xsecMap.items() if key in filepath),("unk",1) )
      self.xsec = xsec
      self.lumi = lumiMap[self.year][0]
      self.scale = self.lumi * xsec / self.total
      self.cutflow_scaled = (self.cutflow * self.scale).astype(int)
      genEventSumw = np.unique(self.get('genEventSumw', library='np')).sum()
      genWeight = self.get('genWeight', library='np') / genEventSumw
      # genWeight = self.get('genWeight', library='np') / self.cutflow_w.to_numpy()[0][0]
      self.genWeight = self.lumi * xsec * genWeight
      self.w_pu = self.get('PUWeight', library='np')
      self.w_pu_up = self.get('PUWeight_up', library='np')
      self.w_pu_down = self.get('PUWeight_down', library='np')
      self.w_puid = self.get('PUIDWeight', library='np')
      self.w_puid_up = self.get('PUIDWeight_up', library='np')
      self.w_puid_down = self.get('PUIDWeight_down', library='np')
      self.w_trigger = self.get('triggerScaleFactor', library='np')
      self.w_trigger_up = self.get('triggerScaleFactorUp', library='np')
      self.w_trigger_down = self.get('triggerScaleFactorDown', library='np')
      self.scale = self.genWeight*self.w_pu*self.w_puid*self.w_trigger
      self.cutflow_scaled = (self.cutflow * self.scale.sum()).astype(int)
      self.nomWeight = self.genWeight*self.w_pu*self.w_puid*self.w_trigger


      self.resolved_mask = ak.all(self.jet_signalId[:,:6] > -1, axis=1)
      if feyn: self.feynnet_efficiency = ak.sum(self.n_h_found[self.resolved_mask] == 3) / ak.sum(self.resolved_mask)

      # for k, v in self.tree.items():
      #    if 'gen' in k:
      #       setattr(self, k, v.array())
      
      self.jet_higgsIdx = (self.jet_signalId) // 2

      self.get_sf_ratios()
      self.init_weights()

   def get_sf_ratios(self):
      """
      The scale factor ratios must first be calculated by running the script located in `scripts/calculate_2d_sf_corrs.py`, which calculates the correction factors using the maxbtag ntuples.
      """
      sf_dir = f'data/{self.year_long}/btag/MX_{self.mx}_MY_{self.my}.root'
      sf_file = uproot.open(sf_dir)

      for key in sf_file.keys():
         if 'cferr' in key: continue
         sf_name = key.split(f'{self.my}_')[-1].split(';')[0]
         
         ratio, n_bins, ht_bins = sf_file[key].to_numpy()
   
         n_jet = self.n_jet.to_numpy()
         ht    = self.get('PFHT', library='np')

         i = get_indices(n_jet, n_bins)
         j = get_indices(ht, ht_bins)
         corr = ratio[i,j]
      
         raw_sf = self.tree[sf_name].array()

         setattr(self, sf_name, corr * raw_sf)
         setattr(self, f'{sf_name}_raw', raw_sf)
      
      self.nomWeight = self.nomWeight * self.bSFshape_central

   def init_weights(self):
      self.w_nominal = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_central*self.w_trigger
      self.w_PUUp = self.genWeight*self.w_pu_up*self.w_puid*self.bSFshape_central*self.w_trigger
      self.w_PUDown = self.genWeight*self.w_pu_down*self.w_puid*self.bSFshape_central*self.w_trigger
      self.w_PUIDUp = self.genWeight*self.w_pu*self.w_puid_up*self.bSFshape_central*self.w_trigger
      self.w_PUIDDown = self.genWeight*self.w_pu*self.w_puid_down*self.bSFshape_central*self.w_trigger
      self.w_triggerUp = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_central*self.w_trigger_up
      self.w_triggerDown = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_central*self.w_trigger_down
      self.w_HFUp = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_up_hf*self.w_trigger
      self.w_HFDown = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_down_hf*self.w_trigger
      self.w_LFUp = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_up_lf*self.w_trigger
      self.w_LFDown = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_down_lf*self.w_trigger
      self.w_LFStats1Up = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_up_lfstats1*self.w_trigger
      self.w_LFStats1Down = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_down_lfstats1*self.w_trigger
      self.w_LFStats2Up = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_up_lfstats2*self.w_trigger
      self.w_LFStats2Down = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_down_lfstats2*self.w_trigger
      self.w_HFStats1Up = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_up_hfstats1*self.w_trigger
      self.w_HFStats1Down = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_down_hfstats1*self.w_trigger
      self.w_HFStats2Up = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_up_hfstats2*self.w_trigger
      self.w_HFStats2Down = self.genWeight*self.w_pu*self.w_puid*self.bSFshape_down_hfstats2*self.w_trigger

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
      
      if title:
         ax.set_title(self.sample)

      if np.issubdtype(bins.dtype, int):
         ax.tick_params(axis='x', which='minor', color='white')
         ax.xaxis.set_ticks(bins)
         # ax.xaxis.set_tick_params(which='minor', bottom=False)
      
      ax.set_ylabel('AU')

      return n

class Data(Tree):
   _is_data = True

   def __init__(self, filepath, cfg=None, treename='sixBtree', feyn=True):
      super().__init__(filepath, treename, cfg, feyn=feyn)

      self.sample = rf"{round(lumiMap[self.year][0]/1000,1)} fb$^{{-1}}$ (13 TeV, {self.year})"

   def train(self):
      print(".. training in validation region")
      self.bdt.train(self, self.vsr_ls_mask, self.vcr_hs_mask, self.vcr_ls_mask, 'vr')
      print(".. training in analysis region")
      self.bdt.train(self, self.asr_ls_mask, self.acr_hs_mask, self.acr_ls_mask, 'ar')

      self.vr_stat_prec = round(1 + np.sqrt(abs(1/self.vsr_weights.sum() - 1/self.asr_weights.sum())), 2)
      err = np.sqrt(np.sum(self.vsr_weights**2))
      self.total_error = np.sqrt(self.vsr_weights.sum() + err**2 + (self.vsr_weights.sum()*self.vrtf)**2)
      self.diff = ak.sum(self.vsr_hs_mask) - ak.sum(self.vsr_weights)
      if self.diff**2 > self.total_error: self.K = np.sqrt(self.diff**2 - self.total_error**2)
      else: self.K = 0
      self.vr_yield_val = round(1 + self.K / ak.sum(self.asr_weights), 2)

      # self.vsr_model_ks_t, self.vsr_model_ks_p = self.ks_test('X_m', self.vsr_ls_mask, self.vsr_hs_mask, self.vsr_weights) # weighted ks test (ls shape with weights)
      # self.vsr_const_ks_t, self.vsr_const_ks_p = self.ks_test('X_m', self.vsr_ls_mask, self.vsr_hs_mask, np.ones_like(self.vsr_weights)) # unweighted ks test (ls shape without weights)
      # self.vcr_model_ks_t, self.vcr_model_ks_p = self.ks_test('X_m', self.vcr_ls_mask, self.vcr_hs_mask, self.vcr_weights)

   # def vr_histo(self, ls_mask, hs_mask, weights, density=False, vcr=False, data_norm=False, ax=None, variable='X_m', bins=None, norm=None):
   #    # ratio = ak.sum(self.vcr_ls_mask) / ak.sum(self.vcr_hs_mask) 
   #    ratio = 1
   #    # fig, axs = plt.subplots()

   #    # if ax is None: 
   #    fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})

   #    # fig.suptitle("Validation Control Region")
   #    var = getattr(self, variable)
   #    var = self.X.m
   #    xlabel = xlabel_dict[variable]
   #    if variable == 'X_m': bins = self.mBins
   #    else: bins = bin_dict[variable]
   #    bins=bins
      
   #    n_ls = np.histogram(var[ls_mask], bins=bins)[0]
   #    n_hs = np.histogram(var[hs_mask], bins=bins)[0]
   #    n_target, n_model, n_ratio, total = Ratio([var[ls_mask], var[hs_mask]], weights=[weights*ratio, None], bins=bins, density=density, axs=axs, labels=['Model', 'Data'], xlabel=r"M$_\mathrm{X}$ [GeV]", ratio_ylabel='Obs/Exp', data_norm=data_norm, norm=norm, total=True)

   #    axs[0].set_ylabel('Events')

   #    axs[1].set_xlabel(xlabel)

   #    if vcr: 
   #       self.bin_ratios = np.nan_to_num(n_hs / n_ls)
   #       # print("bin ratios",self.bin_ratios)
   #    # else: 
   #    #    print(n_ls.sum(), n_hs.sum(), weights.sum(), (n_ls*self.bin_ratios).sum())
   #    sumw2 = []
   #    err = []
   #    for i,n_nominal in enumerate(n_model):#, model_uncertainty_up, model_uncertainty_down)):
   #       low_x = self.X_m[self.vsr_ls_mask] > self.mBins[i]
   #       high_x = self.X_m[self.vsr_ls_mask] <= self.mBins[i+1]
   #       weights = np.sum((self.vsr_weights[low_x & high_x]/total)**2)
   #       sumw2.append(weights)
   #       weights = np.sqrt(weights)
   #       # print(weights, n_nominal)
   #       err.append(weights)
   #       model_uncertainty_up = n_nominal + weights
   #       model_uncertainty_down = n_nominal - weights
   #       # print(model_uncertainty_up,model_uncertainty_down)

   #       # axs[0].fill_between([self.mBins[i], self.mBins[i+1]], model_uncertainty_down, model_uncertainty_up, color='C0', alpha=0.25)
   #       ratio_up   = np.nan_to_num(model_uncertainty_up / n_nominal)
   #       # print(weights, model_uncertainty_up, n_nominal)
   #       ratio_down = np.nan_to_num(model_uncertainty_down / n_nominal)
   #       # print(ratio_up, ratio_down)
   #       axs[1].fill_between([self.mBins[i], self.mBins[i+1]], ratio_down, ratio_up, color='C0', alpha=0.25)

   #    self.VR_sumw2 = np.array((sumw2))
   #    self.VR_err = np.array((err))

   #    return fig, axs, n_target, n_model, n_ratio
      

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
         # fname = f"{savein}/{filepath}_{variable}_before.pdf"
         # fig_before.savefig(saveas, bbox_inches='tight')
         # fname = f"{savein}/{filepath}_{variable}_after.pdf"
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


bins = 31

bin_dict = {
   'pt6bsum' : np.linspace(300,1000,bins),
   'dR6bmin' : np.linspace(0,2,bins),
   'dEta6bmax' : np.linspace(0,4,bins),
   'H1_pt' : np.linspace(0,400,bins),
   'H2_pt' : np.linspace(0,300,bins),
   'H3_pt' : np.linspace(0,300,bins),
   'H1_m' : np.linspace(125,250,bins),
   'H2_m' : np.linspace(125,250,bins),
   'H3_m' : np.linspace(125,250,bins),
   'H3_pt' : np.linspace(0,300,bins),
   'H2_dr' : np.linspace(0,4,bins),
   'H1_dr' : np.linspace(0,4,bins),
   'H3_dr' : np.linspace(0,4,bins),
   'H1_H2_dEta' : np.linspace(0, 5, bins),
   'H2_H3_dEta' : np.linspace(0, 5, bins),
   'H3_H1_dEta' : np.linspace(0, 5, bins),
   'H1_H2_dPhi' : np.linspace(-np.pi, np.pi, bins),
   'H1_costheta' : np.linspace(-1,1,bins),
   'H2_costheta' : np.linspace(-1,1,bins),
   'H3_costheta' : np.linspace(-1,1,bins),
   'Y_m' : np.linspace(250, 900, bins),
   'X_m' : np.linspace(375, 2000, bins),
}

xlabel_dict = {
   'X_m' : r"$M_X$ [GeV]",
   'pt6bsum' : r"$\Sigma_i^{jets} p_{T,i}$ [GeV]",
   'dR6bmin' : r"$min(\Delta R_{jj})$",
   'H1_pt' : r"$H_X \; p_T$ [GeV]",
   'H2_pt' : r"$H_1 \; p_T$ [GeV]",
   'H3_pt' : r"$H_2 \; p_T$ [GeV]",
   'H2_m' : r"$H_1$ mass [GeV]",
   'H1_m' : r"$H_X$ mass [GeV]",
   'H1_dr' : r"$H_X \Delta R_{bb}$",
   'H3_dr' : r"$H_1 \Delta R_{bb}$",
   'H3_dr' : r"$H_2 \Delta R_{bb}$",
   'H2_H3_dEta' : r"$\Delta\eta(H_1,H_2)$",
   'Y_m' : r"$M_Y$ [GeV]"
}

class QCD():

   def __init__(self, trees):

      self.sample = 'QCD'

      model_version = trees[0].model.version

      if model_version == 'new':
         self.H1_m = ak.concatenate([tree.H1.m for tree in trees])
         self.H2_m = ak.concatenate([tree.H2.m for tree in trees])
         self.H3_m = ak.concatenate([tree.H3.m for tree in trees])
         self.rank = ak.concatenate([tree.rank for tree in trees])
         self.ranks = ak.concatenate([tree.ranks for tree in trees])
         self.combo = ak.concatenate([tree.combo for tree in trees])
         self.combos = ak.concatenate([tree.combos for tree in trees])
      elif model_version == 'old':
         self.H1_m = ak.concatenate([tree.HX.m for tree in trees])
         self.H2_m = ak.concatenate([tree.HY1.m for tree in trees])
         self.H3_m = ak.concatenate([tree.HY2.m for tree in trees])
      
      self.X_m = ak.concatenate([tree.X.m for tree in trees])
      self.btag_avg = ak.concatenate([tree.btag_avg for tree in trees])
      self.asr_mask = ak.concatenate([tree.asr_mask for tree in trees])

      self.rank = ak.concatenate([tree.rank for tree in trees])

      genWeights = []
      nomWeights = []

      for tree in trees:
         samp, xsec = next( ((key,value) for key,value in xsecMap.items() if key in tree.filepath),("unk",1) )
         tree.xsec = xsec
         tree.lumi = lumiMap[tree.year][0]

         genEventSumw = ak.sum(np.unique(tree.get('genEventSumw')))
         genWeight = tree.get('genWeight') / genEventSumw
         tree.genWeight = tree.lumi * xsec * genWeight
         tree.w_pu = tree.get('PUWeight')
         tree.w_pu_up = tree.get('PUWeight_up')
         tree.w_pu_down = tree.get('PUWeight_down')
         tree.w_puid = tree.get('PUIDWeight')
         tree.w_puid_up = tree.get('PUIDWeight_up')
         tree.w_puid_down = tree.get('PUIDWeight_down')
         tree.w_trigger = tree.get('triggerScaleFactor')
         tree.w_trigger_up = tree.get('triggerScaleFactorUp')
         tree.w_trigger_down = tree.get('triggerScaleFactorDown')
         tree.nomWeight = tree.genWeight*tree.w_pu*tree.w_puid*tree.w_trigger
         genWeights.append(tree.genWeight)
         nomWeights.append(tree.nomWeight)

      self.genWeight = ak.concatenate(genWeights)
      self.nomWeight = ak.concatenate(nomWeights)
  
class Bkg():
   _is_bkg = True

   def __init__(self, selection='maxbtag_4b', feyn='config/feynnet.cfg'):

      from utils.filelists import get_qcd_list, get_ttbar
      qcdfiles = get_qcd_list(selection)
      ttbarfile = get_ttbar(selection)

      self.ttbar = Tree(ttbarfile, feyn=feyn)
      self.ttbar.sample = 'ttbar'
      self.init_ttbar()
      print(f"Using FeynNet: {self.ttbar.model.model_name}")

      self.qcd_trees = []
      for file in qcdfiles:
         tree = Tree(file, feyn=feyn)
         if tree.nevents > 0: self.qcd_trees.append(tree)
         else: print(f".. skipping {file}")
      self.qcd = QCD(self.qcd_trees)

      n_qcd = ak.sum(self.qcd.nomWeight[self.qcd.asr_mask])
      n_ttbar = ak.sum(self.ttbar.nomWeight[self.ttbar.asr_mask])
      self.sr_ratio_qcd = n_qcd/(n_qcd+n_ttbar)
      self.sr_ratio_ttbar = n_ttbar/(n_qcd+n_ttbar)
      
      n_qcd = ak.sum(self.qcd.nomWeight)
      n_ttbar = ak.sum(self.ttbar.nomWeight)
      self.ratio_qcd = n_qcd/(n_qcd+n_ttbar)
      self.ratio_ttbar = n_ttbar/(n_qcd+n_ttbar)

      # try: self.ttbar_trees = [Tree(file) for file in ttbarfiles]
      # except IndexError: self.ttbar_trees = [Tree(ttbarfiles)]


   def init_ttbar(self):
      samp, xsec = next( ((key,value) for key,value in xsecMap.items() if key in self.ttbar.filepath),("unk",1) )
      self.ttbar.xsec = xsec
      self.ttbar.lumi = lumiMap[self.ttbar.year][0]
      genEventSumw = ak.sum(np.unique(self.ttbar.get('genEventSumw')))
      genWeight = self.ttbar.get('genWeight') / genEventSumw
      self.ttbar.genWeight = self.ttbar.lumi * xsec * genWeight
      self.ttbar.PUWeight = self.ttbar.get('PUWeight')
      self.ttbar.PUWeight_up = self.ttbar.get('PUWeight_up')
      self.ttbar.PUWeight_down = self.ttbar.get('PUWeight_down')
      self.ttbar.PUIDWeight = self.ttbar.get('PUIDWeight')
      self.ttbar.PUIDWeight_up = self.ttbar.get('PUIDWeight_up')
      self.ttbar.PUIDWeight_down = self.ttbar.get('PUIDWeight_down')
      self.ttbar.triggerSF = self.ttbar.get('triggerScaleFactor')
      self.ttbar.triggerSF_up = self.ttbar.get('triggerScaleFactorUp')
      self.ttbar.triggerSF_down = self.ttbar.get('triggerScaleFactorDown')
      self.ttbar.nomWeight = self.ttbar.genWeight*self.ttbar.PUWeight*self.ttbar.PUIDWeight*self.ttbar.triggerSF