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
from utils.analysis.feyn import Model
from utils.xsecUtils import lumiMap, xsecMap
from utils.plotter import latexTitle
from configparser import ConfigParser
from utils.analysis.bdt import BDTRegion

import re
import os, sys 
from colorama import Fore, Style
# import subprocess

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
   ROOT_hist = ROOT.TH1D(h_title,";m_{X} [GeV];Events",nbins-1,array('d',list(m_bins)))
   for i,(val) in enumerate(h_vals):
      ROOT_hist.SetBinContent(i+1, val) 

   ROOT_hist.Draw("hist")
   ROOT_hist.Write()
   fout.Close()

# Parent Class
class Tree():
   def __init__(self, filepath, treename='sixBtree', cfg=None, feyn=True):
      from utils.cutConfig import btagWP
      self.btagWP = btagWP

      self.filepath = filepath
      print(f"{Fore.CYAN}ntuple: {self.filepath}{Style.RESET_ALL}")
      self.filename = self.filepath.split('/')[-2]
      self.treename = treename
      self.year = int([yr for yr in ['2016', '2017', '2018'] if yr in filepath][0])
      self.year_long = re.search("ntuples/(.*?)/", filepath).group(1)
      self.is_signal = 'NMSSM' in filepath
      self.tree = uproot.open(f"{filepath}:{treename}")

      if cfg is None: cfg = f'config/bdt_{self.year_long}.cfg'
      self.read_config(cfg)

      # with uproot.open(f"{filepath}:{treename}") as tree:
      pattern = re.compile('H.+_') # Search for any keys beginning with an 'H' and followed somewhere by a '_'
      for k, v in self.tree.items():
         if re.match(pattern, k) or 'jet' in k or 'gen' in k or 'Y' in k or 'X' in k:
            setattr(self, k, v.array())
      try: 
         with uproot.open(f"{filepath}:h_cutflow_unweighted") as f: cutflow = f
      except:
         with uproot.open(f"{filepath}:h_cutflow") as f: cutflow = f
      self.cutflow_labels = cutflow.axis().labels()
      # self.tree = tree
      self.nevents = int(cutflow.to_numpy()[0][-1])
      self.total = int(cutflow.to_numpy()[0][0])
      self.scale = 1

      self.cutflow = (cutflow.to_numpy()[0]).astype(int)
      self.cutflow_norm = (cutflow.to_numpy()[0]/cutflow.to_numpy()[0][0]*100).astype(int)

      key = str(self.year)
      if key == '2016' and 'preVFP' in self.filepath: key += 'preVFP'
      elif key == '2016': key += 'postVFP'
      self.year_key = key
      self.loose_wp = self.btagWP[key]['Loose']
      self.medium_wp = self.btagWP[key]['Medium']
      self.tight_wp = self.btagWP[key]['Tight']

      if feyn: self.model = Model('new', self) # particles initialized here
      else: self.initialize_bosons()
      self.bdt = BDTRegion(self)
      
      if self.nevents != len(self.n_jet): print("WARNING! Number of events in cutflow does not match number of events in tree!")

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

      HX_b1 = {'pt':sample_particles[0].pt,'eta':sample_particles[0].eta,'phi':sample_particles[0].phi,'m':sample_particles[0].m,'btag':sample_particles[0].btag,'sig_id':sample_particles[0].sig_id,'h_id':sample_particles[0].h_id}
      HX_b2 = {'pt':sample_particles[1].pt,'eta':sample_particles[1].eta,'phi':sample_particles[1].phi,'m':sample_particles[1].m,'btag':sample_particles[1].btag,'sig_id':sample_particles[1].sig_id,'h_id':sample_particles[1].h_id}
      H1_b1 = {'pt':sample_particles[2].pt,'eta':sample_particles[2].eta,'phi':sample_particles[2].phi,'m':sample_particles[2].m,'btag':sample_particles[2].btag,'sig_id':sample_particles[2].sig_id,'h_id':sample_particles[2].h_id}
      H1_b2 = {'pt':sample_particles[3].pt,'eta':sample_particles[3].eta,'phi':sample_particles[3].phi,'m':sample_particles[3].m,'btag':sample_particles[3].btag,'sig_id':sample_particles[3].sig_id,'h_id':sample_particles[3].h_id}
      H2_b1 = {'pt':sample_particles[4].pt,'eta':sample_particles[4].eta,'phi':sample_particles[4].phi,'m':sample_particles[4].m,'btag':sample_particles[4].btag,'sig_id':sample_particles[4].sig_id,'h_id':sample_particles[4].h_id}
      H2_b2 = {'pt':sample_particles[5].pt,'eta':sample_particles[5].eta,'phi':sample_particles[5].phi,'m':sample_particles[5].m,'btag':sample_particles[5].btag,'sig_id':sample_particles[5].sig_id,'h_id':sample_particles[5].h_id}

      self.HX_unregressed = Higgs(HX_b1, HX_b2)

      # self.HX_b1 = self.HX.b1
      # self.HX_b2 = self.HX.b2

      H1 = Higgs(H1_b1, H1_b2)
      H2 = Higgs(H2_b1, H2_b2)

      assert ak.all(self.HX.b1.pt >= self.HX.b2.pt)
      assert ak.all(H1.b1.pt >= H1.b2.pt)
      assert ak.all(H2.b1.pt >= H2.b2.pt)

      self.Y_unregressed = Y(H1, H2)

      self.H1_unregressed = self.Y_unregressed.H1
      self.H2_unregressed = self.Y_unregressed.H2

      assert ak.all(self.H1.pt >= self.H2.pt)

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

class SixB(Tree):

   _is_signal = True

   def __init__(self, filepath, config=None, treename='sixBtree', feyn=True):
      super().__init__(filepath, treename, config, feyn=feyn)
      
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
      if 'Official_NMSSM' in filepath:
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
         self.nomWeight = self.genWeight*self.PUWeight*self.PUIDWeight*self.triggerSF


      self.resolved_mask = ak.all(self.jet_signalId[:,:6] > -1, axis=1)
      # assert np.array_equal(self.gnn_resolved_mask, self.resolved_mask)

      self.H_b_h_id = np.column_stack((self.jet_signalId[:,:6]+1))//3


      hx_possible = ak.sum(self.H_b_h_id == 0, axis=1) == 2
      h1_possible = ak.sum(self.H_b_h_id == 1, axis=1) == 2
      h2_possible = ak.sum(self.H_b_h_id == 2, axis=1) == 2
      self.n_h_possible = hx_possible*1 + h1_possible*1 + h2_possible*1


      # for k, v in self.tree.items():
      #    if 'gen' in k:
      #       setattr(self, k, v.array())
      
      self.jet_higgsIdx = (self.jet_signalId) // 2

      # print("Calculating SF correction factors")
      if 'Official' in self.filepath: 
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
         # print(sf_name)
         
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
      self.w_nominal = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_central*self.triggerSF
      self.w_PUUp = self.genWeight*self.PUWeight_up*self.PUIDWeight*self.bSFshape_central*self.triggerSF
      self.w_PUDown = self.genWeight*self.PUWeight_down*self.PUIDWeight*self.bSFshape_central*self.triggerSF
      self.w_PUIDUp = self.genWeight*self.PUWeight*self.PUIDWeight_up*self.bSFshape_central*self.triggerSF
      self.w_PUIDDown = self.genWeight*self.PUWeight*self.PUIDWeight_down*self.bSFshape_central*self.triggerSF
      self.w_triggerUp = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_central*self.triggerSF_up
      self.w_triggerDown = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_central*self.triggerSF_down
      self.w_HFUp = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_up_hf*self.triggerSF
      self.w_HFDown = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_down_hf*self.triggerSF
      self.w_LFUp = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_up_lf*self.triggerSF
      self.w_LFDown = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_down_lf*self.triggerSF
      self.w_LFStats1Up = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_up_lfstats1*self.triggerSF
      self.w_LFStats1Down = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_down_lfstats1*self.triggerSF
      self.w_LFStats2Up = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_up_lfstats2*self.triggerSF
      self.w_LFStats2Down = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_down_lfstats2*self.triggerSF
      self.w_HFStats1Up = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_up_hfstats1*self.triggerSF
      self.w_HFStats1Down = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_down_hfstats1*self.triggerSF
      self.w_HFStats2Up = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_up_hfstats2*self.triggerSF
      self.w_HFStats2Down = self.genWeight*self.PUWeight*self.PUIDWeight*self.bSFshape_down_hfstats2*self.triggerSF

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

