"""
Store ROOT events in a standard and convenient way.

Classes:
    Signal
"""

from utils import *
from utils.cutConfig import jet_btagWP
from utils.varUtils import *
from utils.plotter import latexTitle, Hist
from utils.useCMSstyle import *
plt.style.use(CMS)
from .particle import Particle
from awkward.highlevel import Array

# Standard library imports
from array import array
# import awkward0 as ak0
from colorama import Fore
from hep_ml import reweight
import re
import ROOT
ROOT.gROOT.SetBatch(True)
import sys 
import uproot
import pandas as pd
# from hep_ml.metrics_utils import ks_2samp_weighted
from scipy.stats import kstwobign

njet_bins = np.arange(8)
id_bins = np.arange(-1, 7)
pt_bins = np.linspace(0, 500, 100)
score_bins = np.linspace(0,1.01,102)

nbins = 40
m_bins = np.linspace(375, 1150, nbins)
x_X_m = (m_bins[:-1] + m_bins[1:]) / 2

def ROOTHist(n, mass_name, title):
      fout = ROOT.TFile(f"ml/gnn/{mass_name}.root","recreate")
      fout.cd()

      canvas = ROOT.TCanvas('c1','c1', 600, 600)
      canvas.SetFrameLineWidth(3)
      canvas.Draw()

      h_title = title
      ROOT_hist = ROOT.TH1D(h_title,";m_{X} [GeV];Events",nbins-1,array('d',list(m_bins)))
      for i,(val) in enumerate(n):
         ROOT_hist.SetBinContent(i+1, val) 

      ROOT_hist.Draw("hist")
      ROOT_hist.Write()
      fout.Close()



# def presels(tree, pt=20, eta=2.5, jetid=6, puid=6):
#    pt_mask = ak.sum(tree.jet_pt > pt, axis=1) >= 6
#    eta_mask = ak.sum(abs(tree.jet_eta) < eta, axis=1) >= 6
#    jetid_mask = ak.sum(tree.jet_id == jetid, axis=1) >= 6
#    pt_under50 = tree.jet_pt < 50
#    puid_mask = ak.sum(tree.jet_puid[pt_under50] == puid, axis=1) + ak.sum(~pt_under50, axis=1) >= 6

#    presel_mask = pt_mask & eta_mask & jetid_mask & puid_mask
#    return tree[presel_mask]




from configparser import ConfigParser
# Parent Class
class Tree():

   def __init__(self, filename, treename='sixBtree', cfg_file=None, year=2018, selection='3322'):

      with uproot.open(f"{filename}:{treename}") as tree:
         pattern = re.compile('H.+_') # Search for any keys beginning with an 'H' and followed somewhere by a '_'
         for k, v in tree.items():
            if re.match(pattern, k) or 'jet' in k or 'gen' in k or 'Y' in k or 'X' in k:
               setattr(self, k, v.array())
            
      cutflow = uproot.open(f"{filename}:h_cutflow_unweighted")
      # self.tree = tree
      self.nevents = int(cutflow.to_numpy()[0][-1])
      self.total = int(cutflow.to_numpy()[0][0])
      self.scale = 1

      self.cutflow = (cutflow.to_numpy()[0]).astype(int)
      self.cutflow_norm = (cutflow.to_numpy()[0]/cutflow.to_numpy()[0][0]*100).astype(int)

      self.initialize_vars()

      self.cfg = cfg_file

      self.spherical_region()

   def keys(self):
      return self.tree.keys()

   def find_key(self, start):
      for key in self.keys():
         if start in key : print(key)

   def get(self, key, library='ak'):
      """Returns the key.
      Use case: Sometimes keys are accessed directly, e.g. tree.key, but other times, the key may be accessed from a list of keys, e.g. tree.get(['key']). This function handles the latter case.
      """
      try: 
         getattr(self, key)
         if library=='np' and not isinstance(arr, np.ndarray): arr = arr.to_numpy()
      except:
         return self.tree[key].array(library=library)
    
   def np(self, key):
      """Returns the key as a numpy array."""
      np_arr = self.get(key, library='np')
      if not isinstance(np_arr, np.ndarray): np_arr = np_arr.to_numpy()
      return np_arr

   def initialize_vars(self):
      """Initialize variables that don't exist in the original ROOT tree."""

      # allow for various b tag cuts
      # jet_btag = self.tree['jet_btag'].array()
      # self.btag_mask = ak.count(jet_btag, axis=1) > 0
      # for i,s in enumerate(selection):
      #    btag_cut = jet_btagWP[int(s)]
      #    jet_pass_criteria = jet_btag[:,i] > btag_cut
      #    self.btag_mask = self.btag_mask & jet_pass_criteria

      # pattern = re.compile('H.+_') # Search for any keys beginning with an 'H' and followed somewhere by a '_'
      # for k, v in self.tree.items():
      #    if re.match(pattern, k) or 'jet' in k or 'Y' in k or 'X' in k or:
      #       setattr(self, k, v.array())
      #       setattr(self, k, v.array()[self.btag_mask])

      HX_b1 = Particle(self, 'HX_b1')
      HX_b2 = Particle(self, 'HX_b2')
      H1_b1 = Particle(self, 'H1_b1')
      H1_b2 = Particle(self, 'H1_b2')
      H2_b1 = Particle(self, 'H2_b1')
      H2_b2 = Particle(self, 'H2_b2')

      HX = Particle(self, 'HX')
      H1 = Particle(self, 'H1')
      H2 = Particle(self, 'H2')

      Y = Particle(self, 'Y')

      X = Particle(self, 'X')

      bs = [HX_b1, HX_b2, H1_b1, H1_b2, H2_b1, H2_b2]
      pair1 = [HX_b1]*5 + [HX_b2]*4 + [H1_b1]*3 + [H1_b2]*2 + [H2_b1]
      pair2 = bs[1:] + bs[2:] + bs[3:] + bs[4:] + [bs[-1]]

      dR6b = []
      dEta6b = []
      for b1, b2 in zip(pair1, pair2):
         dR6b.append(b1.deltaR(b2).to_numpy())
         dEta6b.append(abs(b1.deltaEta(b2).to_numpy()))
      dR6b = np.column_stack(dR6b)
      dEta6b = np.column_stack(dEta6b)
      self.dR6bmin = dR6b.min(axis=1)
      self.dEta6bmax = dEta6b.max(axis=1)

      self.pt6bsum = HX_b1.pt + HX_b2.pt + H1_b1.pt + H1_b2.pt + H2_b1.pt + H2_b2.pt

      # self.HX_dr = HX_b1.deltaR(HX_b2)
      # self.H1_dr = H1_b1.deltaR(H1_b2)
      # self.H2_dr = H2_b1.deltaR(H2_b2)

      self.HX_H1_dEta = abs(HX.deltaEta(H1))
      self.H1_H2_dEta = abs(H1.deltaEta(H2))
      self.H2_HX_dEta = abs(H2.deltaEta(HX))

      self.HX_H1_dPhi = HX.deltaPhi(H1)
      self.H1_H2_dPhi = H1.deltaPhi(H2)
      self.H2_HX_dPhi = H2.deltaPhi(HX)

      self.HX_H1_dr = HX.deltaR(H1)
      self.HX_H2_dr = H2.deltaR(HX)
      self.H1_H2_dr = H1.deltaR(H2)
      
      self.Y_HX_dr = Y.deltaR(HX)

      self.HX_costheta = abs(np.cos(HX.P4.theta))
      self.H1_costheta = abs(np.cos(H1.P4.theta))
      self.H2_costheta = abs(np.cos(H2.P4.theta))

      self.HX_H1_dr = HX.deltaR(H1)
      self.H1_H2_dr = H2.deltaR(H1)
      self.H2_HX_dr = HX.deltaR(H2)

      # X = HX + H1 + H2
      self.X_m = X.m

      self.H_j_btag = np.column_stack((
         self.HX_b1_btag.to_numpy(),
         self.HX_b2_btag.to_numpy(),
         self.H1_b1_btag.to_numpy(),
         self.H1_b2_btag.to_numpy(),
         self.H2_b1_btag.to_numpy(),
         self.H2_b2_btag.to_numpy()
      ))

      self.n_H_jet_tight = np.sum(self.H_j_btag >= jet_btagWP[3], axis=1)
      self.n_H_jet_medium = np.sum(self.H_j_btag >= jet_btagWP[2], axis=1)
      self.n_H_jet_loose = np.sum(self.H_j_btag >= jet_btagWP[1], axis=1)

      self.btag_avg = np.average(self.H_j_btag, axis=1)



   def spherical_region(self):

      self.config = ConfigParser()
      self.config.optionxform = str
      self.config.read(self.cfg)
      self.config = self.config

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
      self.AR_center = float(self.config['spherical']['ARcenter'])
      self.SR_edge   = float(self.config['spherical']['rInner'])
      self.CR_edge   = float(self.config['spherical']['rOuter'])

      self.VR_center = int(self.AR_center + (self.SR_edge + self.CR_edge)/np.sqrt(2))

      higgs = ['HX_m', 'H1_m', 'H2_m']

      deltaM = np.column_stack(([getattr(self, mH).to_numpy() - self.AR_center for mH in higgs]))
      deltaM = deltaM * deltaM
      deltaM = deltaM.sum(axis=1)
      AR_deltaM = np.sqrt(deltaM)
      self.A_SR_mask = AR_deltaM <= self.SR_edge # Analysis SR
      if self._is_signal: self.A_SR_avgbtag = self.btag_avg[self.A_SR_mask]
      self.A_CR_mask = (AR_deltaM > self.SR_edge) & (AR_deltaM <= self.CR_edge) # Analysis CR
      if not self._is_signal: self.A_CR_avgbtag = self.btag_avg[self.A_CR_mask]

      VR_deltaM = np.column_stack(([abs(getattr(self, mH).to_numpy() - self.VR_center) for mH in higgs]))
      VR_deltaM = VR_deltaM * VR_deltaM
      VR_deltaM = VR_deltaM.sum(axis=1)
      VR_deltaM = np.sqrt(VR_deltaM)
      self.V_SR_mask = VR_deltaM <= self.SR_edge # Validation SR
      self.V_CR_mask = (VR_deltaM > self.SR_edge) & (VR_deltaM <= self.CR_edge) # Validation CR

      score_cut = float(self.config['score']['threshold'])
      self.ls_mask = self.btag_avg < score_cut # ls
      self.hs_mask = self.btag_avg >= score_cut # hs


      # b_cut = float(config['score']['n'])
      # self.nloose_b = ak.sum(self.get('jet_btag') > 0.0490, axis=1)
      # self.nmedium_b = ak.sum(self.get('jet_btag') > 0.2783, axis=1)
      # ls_mask = self.nmedium_b < b_cut # ls
      # hs_mask = self.nmedium_b >= b_cut # hs

      self.A_CRls_mask = self.A_CR_mask & self.ls_mask
      self.A_CRhs_mask = self.A_CR_mask & self.hs_mask
      self.A_SRls_mask = self.A_SR_mask & self.ls_mask
      self.A_SRhs_mask = self.A_SR_mask & self.hs_mask
      if not self._is_signal:
         self.blind_mask = ~self.A_SRhs_mask
         self.A_SRhs_mask = np.zeros_like(self.A_SR_mask)
      # else: self.A_CR_avgbtag = self.btag_avg[self.V_CR_mask]

      self.V_CRls_mask = self.V_CR_mask & self.ls_mask
      self.V_CRhs_mask = self.V_CR_mask & self.hs_mask
      self.V_SRls_mask = self.V_SR_mask & self.ls_mask
      self.V_SRhs_mask = self.V_SR_mask & self.hs_mask




class SixB(Tree):

   _is_signal = True

   def __init__(self, filename, config='config/bdt_params.cfg', treename='sixBtree', year=2018):
      super().__init__(filename, treename, config, year)

      print(filename)
      self.mx = int(re.search('MX_.+MY', filename).group().split('_')[1])
      self.my = int(re.search('MY.+/', filename).group().split('_')[1].split('/')[0])
      # self.filename = re.search('NMSSM_.+/', filename).group()[:-1]
      self.sample = latexTitle(self.mx, self.my)
      self.mxmy = self.sample.replace('$','').replace('_','').replace('= ','_').replace(', ','_').replace(' GeV','')

      samp, xsec = next( ((key,value) for key,value in xsecMap.items() if key in filename),("unk",1) )
      self.xsec = xsec
      self.lumi = lumiMap[year][0]
      self.scale = self.lumi * xsec / self.total

      self.cutflow_scaled = (self.cutflow * self.scale).astype(int)

      # for k, v in self.tree.items():
      #    if 'gen' in k:
      #       setattr(self, k, v.array())
      
      self.jet_higgsIdx = (self.jet_signalId) // 2

   # def get_m_avgb_hist(self):
   #    m_bins = np.linspace(50,200,100)
   #    max_score = 1 + 1e-6
   #    score_bins = np.linspace(0,max_score,100)

   #    fig, ax = plt.subplots()

   #    n, xe, ye, im = Hist2d(self.HX_m, self.btag_avg, bins=(m_bins, score_bins), ax=ax)
   #    ax.set_xlabel(r"{H_X} mass [GeV]")
   #    ax.set_ylabel("Average b tag score of six jets")

   #    fig.colorbar(im, ax=ax)

   #    return fig, ax

   def sr_hist(self):
      fig, ax = plt.subplots()
      n = Hist(self.X_m[self.A_SRhs_mask], bins=self.mBins, ax=ax, density=False, weights=self.scale, zorder=2)
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

data_lumi = {
   2018 : 59.7
}

class Data(Tree):

   _is_signal = False

   def __init__(self, filename, config='config/bdt_params.cfg', treename='sixBtree', year=2018):
      super().__init__(filename, treename, config, year)

      self.sample = rf"{data_lumi[year]} fb$^{{-1}}$ (13 TeV, 2018)"
      self.set_bdt_params()

   def set_bdt_params(self, cfg=None):

      if cfg is not None:
         config = ConfigParser()
         config.optionxform = str
         config.read(self.cfg)
         self.config = config

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

   def set_variables(self, var_list):
      self.variables = var_list

   def get_df(self, mask, variables):
      features = {}
      for var in variables:
         features[var] = abs(getattr(self, var)[mask])
      df = pd.DataFrame(features)
      return df

   def train_ar(self):
      # print(".. initializing transfer factor")
      self.AR_TF = sum(self.A_CRhs_mask)/sum(self.A_CRls_mask)
      ls_weights = np.ones(ak.sum(self.A_CRls_mask))*self.AR_TF
      hs_weights = np.ones(ak.sum([self.A_CRhs_mask]))

      # print(".. initializing dataframes of variables")
      AR_df_ls = self.get_df(self.A_CRls_mask, self.variables)
      AR_df_hs = self.get_df(self.A_CRhs_mask, self.variables)

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
      self.AR_reweighter = reweighter

      # print(".. predicting AR hs weights")
      AR_df_ls = self.get_df(self.A_SRls_mask, self.variables)
      initial_weights = np.ones(ak.sum(self.A_SRls_mask))*self.AR_TF

      self.AR_ls_weights = self.AR_reweighter.predict_weights(AR_df_ls,initial_weights,lambda x: np.mean(x, axis=0))
      
      # self.AR_ls_weights = self.AR_reweighter.predict_weights(AR_df_ls,initial_weights,lambda x: np.mean(x, axis=0))

   def train_vr(self):
      
      # print(".. initializing transfer factor")
      self.VR_TF = sum(self.V_CRhs_mask)/sum(self.V_CRls_mask)
      ls_weights = np.ones(ak.sum(self.V_CRls_mask))*self.VR_TF
      hs_weights = np.ones(ak.sum([self.V_CRhs_mask]))

      # print(".. initializing dataframes of variables")
      df_ls = self.get_df(self.V_CRls_mask, self.variables)
      df_hs = self.get_df(self.V_CRhs_mask, self.variables)

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
      self.VR_reweighter = reweighter

      # print(".. predicting VR hs weights")
      V_cr_df_ls = self.get_df(self.V_CRls_mask, self.variables)
      V_cr_initial_weights = np.ones(ak.sum(self.V_CRls_mask))*self.VR_TF
      self.V_CR_ls_weights = self.VR_reweighter.predict_weights(V_cr_df_ls,V_cr_initial_weights,lambda x: np.mean(x, axis=0))

      V_sr_df_ls = self.get_df(self.V_SRls_mask, self.variables)
      V_sr_initial_weights = np.ones(ak.sum(self.V_SRls_mask))*self.VR_TF
      self.V_SR_ls_weights = self.VR_reweighter.predict_weights(V_sr_df_ls,V_sr_initial_weights,lambda x: np.mean(x, axis=0))

   def train(self):
      self.config.read(self.cfg)
      print(".. training in validation region")
      self.train_vr()
      print()
      print(".. training in analysis region")
      self.train_ar()
      print()

      self.v_sr_model_ks_t, self.v_sr_model_ks_p = self.ks_test('X_m', self.V_SRls_mask, self.V_SRhs_mask, self.V_SR_ls_weights) # weighted ks test (ls shape with weights)
      self.v_sr_const_ks_t, self.v_sr_const_ks_p = self.ks_test('X_m', self.V_SRls_mask, self.V_SRhs_mask, np.ones_like(self.V_SR_ls_weights)) # unweighted ks test (ls shape without weights)
      self.v_cr_model_ks_t, self.v_cr_model_ks_p = self.ks_test('X_m', self.V_CRls_mask, self.V_CRhs_mask, self.V_CR_ls_weights)



   def vr_hist(self, ls_mask, hs_mask, weights, density=False, vcr=False, data_norm=False, ax=None, variable='X_m', bins=None, norm=None):
      # ratio = ak.sum(self.V_CRls_mask) / ak.sum(self.V_CRhs_mask) 
      ratio = 1
      # fig, axs = plt.subplots()

      # if ax is None: 
      fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})

      # fig.suptitle("Validation Control Region")
      var = getattr(self, variable)
      xlabel = xlabel_dict[variable]
      if variable == 'X_m': bins = self.mBins
      else: bins = bin_dict[variable]
      bins=bins
      
      n_ls = np.histogram(var[ls_mask], bins=bins)[0]
      n_hs = np.histogram(var[hs_mask], bins=bins)[0]
      # print(n_ls.sum(), n_hs.sum(), weights.sum())
      # print(np.histogram(var[ls_mask], bins=bins, weights=weights)[0])
      # print(weights, weights.shape)
      n_target, n_model, n_ratio = Ratio([var[ls_mask], var[hs_mask]], weights=[weights*ratio, None], bins=bins, density=density, axs=axs, labels=['Model', 'Data'], xlabel=r"M$_\mathrm{X}$ [GeV]", ratio_ylabel='Obs/Exp', data_norm=data_norm, norm=norm)

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
         low_x = self.X_m[self.V_SRls_mask] > self.mBins[i]
         high_x = self.X_m[self.V_SRls_mask] <= self.mBins[i+1]
         weights = np.sum(self.V_SR_ls_weights[low_x & high_x]**2)
         sumw2.append(weights)
         weights = np.sqrt(weights)
         err.append(weights)
         # model_uncertainty_up = n_nominal + weights
         # model_uncertainty_down = n_nominal - weights

         # axs[0].fill_between([self.mBins[i], self.mBins[i+1]], model_uncertainty_down, model_uncertainty_up, color='C0', alpha=0.25)
         # ratio_up = model_uncertainty_up / n_nominal
         # ratio_down = model_uncertainty_down / n_nominal
         # # print(ratio_down)
         # axs[1].fill_between([self.mBins[i], self.mBins[i+1]], ratio_down, ratio_up, color='C0', alpha=0.25)

      self.VR_sumw2 = np.array((sumw2))
      self.VR_err = np.array((err))

      return fig, axs, n_target, n_model, n_ratio

   def ks_test(self, variable, ls_mask, hs_mask, weights):
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
      ks_probability = int(kstwobign.sf(ks_statistic)*100)

      return ks_statistic, ks_probability
      
   def v_sr_hist(self):
      fig, axs, n_target, n_model, n_ratio = self.vr_hist(self.V_SRls_mask, self.V_SRhs_mask, self.V_SR_ls_weights, density=False)
      axs[0].set_title('Validation Signal Region')
      return fig, axs, n_target, n_model
   
   def v_cr_hist(self):
      fig, axs, n_target, n_model, n_ratio = self.vr_hist(self.V_CRls_mask, self.V_CRhs_mask, self.V_CR_ls_weights, density=False, vcr=True)
      axs[0].set_title('Validation Control Region')
      return fig, axs, n_target, n_model

   def before_after(self, savedir=None, variable='X_m'):
      # xlabel = xlabel_dict[variable]

      fig, axs, n_obs, n_unweighted = self.vr_hist(self.V_SRls_mask, self.V_SRhs_mask, np.ones_like(self.V_SR_ls_weights)/sum(self.V_SRls_mask), data_norm=True, variable=variable)
      axs[0].set_title('Validation Signal Region - Before Applying BDT Weights', fontsize=18)

      if savedir is not None: fig.savefig(f"{savedir}/{variable}_vsr_before_bdt.pdf")

      fig, axs, n_obs, n_weighted = self.vr_hist(self.V_SRls_mask, self.V_SRhs_mask, self.V_SR_ls_weights/self.V_SR_ls_weights.sum(), data_norm=True, variable=variable)
      axs[0].set_title('Validation Signal Region - After Applying BDT Weights', fontsize=18)
      if savedir is not None: fig.savefig(f"{savedir}/{variable}_vsr_after_bdt.pdf")

      # fig.savefig(f"plots/model_VCR.pdf", bbox_inches='tight')

   def sr_hist(self):
      from matplotlib.lines import Line2D
      from matplotlib.patches import Patch

      fig, axs = plt.subplots(nrows=2,  gridspec_kw={'height_ratios':[4,1]})

      n_model_SR_hs = Hist(self.X_m[self.A_SRls_mask], weights=self.AR_ls_weights, bins=self.mBins, ax=axs[0], label='A_SR', density=False)

      model_uncertainty = np.sqrt(n_model_SR_hs)

      sumw2 = []
      err = []
      for i,n_nominal in enumerate(n_model_SR_hs):#, model_uncertainty_up, model_uncertainty_down)):
         low_x = self.X_m[self.A_SRls_mask] > self.mBins[i]
         high_x = self.X_m[self.A_SRls_mask] <= self.mBins[i+1]
         weights = np.sum(self.AR_ls_weights[low_x & high_x]**2)
         sumw2.append(weights)
         weights = np.sqrt(weights)
         err.append(weights)
         model_uncertainty_up = n_nominal + weights
         model_uncertainty_down = n_nominal - weights

         axs[0].fill_between([self.mBins[i], self.mBins[i+1]], model_uncertainty_down, model_uncertainty_up, color='C0', alpha=0.25)
         ratio_up = model_uncertainty_up / n_nominal
         ratio_down = model_uncertainty_down / n_nominal
         # print(ratio_down)
         axs[1].fill_between([self.mBins[i], self.mBins[i+1]], ratio_down, ratio_up, color='C0', alpha=0.25)

      self.sumw2 = np.array((sumw2))
      self.err = np.array((err))

      model_nominal = Line2D([0], [0], color='C0', lw=2, label='Bkg Model')

      handles = [model_nominal, Patch(facecolor='C0', alpha=0.25, label='Bkg Uncertainty')]
      
      axs[0].legend(handles=handles)

      axs[0].set_ylabel('Events')
      axs[1].set_ylabel('Uncertainty')

      axs[1].plot([self.mBins[0], self.mBins[-1]], [1,1], color='gray', linestyle='--')
      axs[1].set_xlabel(r"$M_X$ [GeV]")
      
      return fig, axs, n_model_SR_hs

bins = 31

bin_dict = {
   'pt6bsum' : np.linspace(300,1000,bins),
   'dR6bmin' : np.linspace(0,4,bins),
   'dEta6bmax' : np.linspace(0,3,bins),
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




from itertools import combinations
combos = list(combinations(np.arange(6),2))
N = len(combos)
combo_dict = {}

indices = []
total = 0
for i,combo1 in enumerate(combos):
    combo1 = np.array(combo1)
    for j in range(i+1, N):
        combo2 = np.array(combos[j])
        # print(combo,combos[j])
        if combo1[0] in combo2 or combo1[1] in combo2: continue
        for k in range(j+1, N):
            combo3 = np.array(combos[k])
            if combo2[0] in combo3 or combo2[1] in combo3: continue
            if combo1[0] in combo3 or combo1[1] in combo3: continue
            indices.append(np.concatenate((combo1, combo2, combo3)))
            combo_dict[total] = np.concatenate((combo1, combo2, combo3))
            # print(f"combo_dict[{total}] = {combo_dict[total]}")
            # combo_dict[total] = 
            total += 1
        # break
    # break
indices = np.row_stack([ind for ind in indices])

class GNN(Tree):
   """
   A class to handle the output from the gnn designed to choose the best jet pairs.
   """

   model = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/trih_ranker_mp/20221212_ranger_lr0.0047_batch512/predict_output/trih_ranker_mp_NMSSM_XYH_YToHH_6b_MX_700_MY_400_ntuple.awkd'
   
   def __init__(self, filename, model=None):

      if model is not None: self.model = model
      self.tree = SixB(filename)

      import awkward0 as ak0
      with ak0.load(self.model) as f_awk:
         scores = ak.unflatten(f_awk['scores'], np.repeat(15, self.tree.nevents)).to_numpy()
         maxscore = f_awk['maxscore']
         maxcomb = f_awk['maxcomb']

      combo_arr = np.asarray(([combo_dict[score] for score in scores.argmax(axis=1)]))
      combo_arr = ak.from_numpy(combo_arr)
      combo_arr = ak.unflatten(ak.flatten(combo_arr), ak.ones_like(combo_arr[:,0])*6)
      self.combo_arr = combo_arr

      j1_index = combo_arr[:,::2]
      j2_index = combo_arr[:,1::2]

      h_j1_id = (self.tree.jet_signalId[j1_index]+2)//2
      h_j2_id = (self.tree.jet_signalId[j2_index]+2)//2
      self.h_id = ak.where(h_j1_id == h_j2_id, h_j1_id, 0)[ak.sum(self.tree.jet_signalId[:,:6] > -1, axis=1) == 6]

      particles = []
      for i in range(6):
         particles.append(Particle(kin_dict={
            'pt' : self.tree.jet_ptRegressed[combo_arr][:,i],
            'eta' : self.tree.jet_eta[combo_arr][:,i],
            'phi' : self.tree.jet_phi[combo_arr][:,i],
            'm' : self.tree.jet_m[combo_arr][:,i]
         }))

      self.H1 = particles[0] + particles[1]
      self.H2 = particles[2] + particles[3]
      self.H3 = particles[4] + particles[5]


   def get_eff_hist(self):
      fig, ax = plt.subplots()

      self.tree.hist(ak.sum(self.h_id>0, axis=-1), bins=np.arange(5), density=True, ax=ax)
      ax.set_xlabel('Number of Accurate Higgs Pairs')

   def spherical_region(self, cfg='config/bdt_params.cfg'):

      self.config = ConfigParser()
      self.config.optionxform = str
      self.config.read(cfg)
      self.config = self.config

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
      self.AR_center = float(self.config['spherical']['ARcenter'])
      self.SR_edge   = float(self.config['spherical']['rInner'])
      self.CR_edge   = float(self.config['spherical']['rOuter'])

      self.VR_center = int(self.AR_center + (self.SR_edge + self.CR_edge)/np.sqrt(2))

      higgs = [self.H1.m, self.H2.m, self.H3.m]

      deltaM = np.column_stack(([mh.to_numpy() - self.AR_center for mh in higgs]))
      deltaM = deltaM * deltaM
      deltaM = deltaM.sum(axis=1)
      AR_deltaM = np.sqrt(deltaM)
      self.A_SR_mask = AR_deltaM <= self.SR_edge # Analysis SR
      # if self._is_signal: self.A_SR_avgbtag = self.btag_avg[self.A_SR_mask]
      self.A_SR_avgbtag = self.tree.btag_avg[self.A_SR_mask]
      self.A_CR_mask = (AR_deltaM > self.SR_edge) & (AR_deltaM <= self.CR_edge) # Analysis CR
      # if not self._is_signal: self.A_CR_avgbtag = self.btag_avg[self.A_CR_mask]

      VR_deltaM = np.column_stack(([mh.to_numpy() - self.VR_center for mh in higgs]))
      VR_deltaM = VR_deltaM * VR_deltaM
      VR_deltaM = VR_deltaM.sum(axis=1)
      VR_deltaM = np.sqrt(VR_deltaM)
      self.V_SR_mask = VR_deltaM <= self.SR_edge # Validation SR
      self.V_CR_mask = (VR_deltaM > self.SR_edge) & (VR_deltaM <= self.CR_edge) # Validation CR

      score_cut = float(self.config['score']['threshold'])
      self.ls_mask = self.tree.btag_avg < score_cut # ls
      self.hs_mask = self.tree.btag_avg >= score_cut # hs


      # b_cut = float(config['score']['n'])
      # self.nloose_b = ak.sum(self.get('jet_btag') > 0.0490, axis=1)
      # self.nmedium_b = ak.sum(self.get('jet_btag') > 0.2783, axis=1)
      # ls_mask = self.nmedium_b < b_cut # ls
      # hs_mask = self.nmedium_b >= b_cut # hs

      self.A_CRls_mask = self.A_CR_mask & self.ls_mask
      self.A_CRhs_mask = self.A_CR_mask & self.hs_mask
      self.A_SRls_mask = self.A_SR_mask & self.ls_mask
      self.A_SRhs_mask = self.A_SR_mask & self.hs_mask
      # if not self._is_signal:
      #    self.blind_mask = ~self.A_SRhs_mask
      #    self.A_SRhs_mask = np.zeros_like(self.A_SR_mask)
      # else: self.A_CR_avgbtag = self.btag_avg[self.V_CR_mask]

      self.V_CRls_mask = self.V_CR_mask & self.ls_mask
      self.V_CRhs_mask = self.V_CR_mask & self.hs_mask
      self.V_SRls_mask = self.V_SR_mask & self.ls_mask
      self.V_SRhs_mask = self.V_SR_mask & self.hs_mask
