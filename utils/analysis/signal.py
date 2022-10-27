"""
Store ROOT events in a standard and convenient way.

Classes:
    Signal
"""

from cmath import nan
from utils import *
from utils.varUtils import *
from utils.plotter import latexTitle, Hist
from utils.useCMSstyle import *
plt.style.use(CMS)
from .particle import Particle

# Standard library imports
from array import array
import awkward0 as ak0
from colorama import Fore
from hep_ml import reweight
import re
import ROOT
ROOT.gROOT.SetBatch(True)
import sys 
import uproot
import pandas as pd
from hep_ml.metrics_utils import ks_2samp_weighted

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

class Signal():
   """A class for handling TTrees from recent skims, which output a single branch for each b jet kinematic. (Older skims output an array of jet kinematics.)
   
   Attributes
   ----------
   filename : str
      location of ROOT file
   treename : str
      name of tree to access in ROOT file
   year : int
      year for which to access run conditions
   """

   _rect_region_bool = False
   _sphere_region_bool = False

   def __init__(self, filename, treename='sixBtree', year=2018, presel=False, gen=False, b_cutflow=False):
      if 'NMSSM' in filename:
         if 'presel' in filename: presel=True
         if 'gen' in filename: gen=True
         print(filename)
         self.filename = re.search('NMSSM_.+/', filename).group()[:-1]
      tree = uproot.open(f"{filename}:{treename}")
      self.tree = tree

      # Search for any keys beginning with an 'H' and followed somewhere by a '_'
      pattern = re.compile('H.+_')
      for k, v in tree.items():
         if re.match(pattern, k):
               setattr(self, k, v.array())

      self.nevents = len(tree['Run'].array())

      if 'NMSSM' in filename:
         cutflow = uproot.open(f"{filename}:h_cutflow_unweighted")
         # save total number of events for scaling purposes
         total = cutflow.to_numpy()[0][0]
         self.scaled_total = total
         samp, xsec = next( ((key,value) for key,value in xsecMap.items() if key in filename),("unk",1) )
         self.sample = latexTitle(filename)
         self.xsec = xsec
         self.lumi = lumiMap[year][0]
         self.scale = self.lumi*xsec/total
         self.cutflow = (cutflow.to_numpy()[0]*self.scale).astype(int)
         self.cutflow_norm = (cutflow.to_numpy()[0]/cutflow.to_numpy()[0][0]).astype(int)
         self.mxmy = self.sample.replace('$','').replace('_','').replace('= ','_').replace(', ','_').replace(' GeV','')
         self._isdata = False
      else:
         self.scale = 1
         self._isdata = True
         self.cutflow = uproot.open(f"{filename}:h_cutflow_unweighted").to_numpy()[0]

      for k, v in tree.items():
         if 'jet' in k or 'gen' in k:
            setattr(self, k, v.array())
      self.jet_higgsIdx = (self.jet_signalId) // 2
      
      if 'dnn' in filename: self.evt_score = tree['b_6j_score'].array()

      if not presel and not gen:
         # self.tree['jet_higgsIdx'] = self.jet_higgsIdx
         jets = ['HX_b1_', 'HX_b2_', 'H1_b1_', 'H1_b2_', 'H2_b1_', 'H2_b2_']
         try: self.btag_avg = np.column_stack(([self.get(jet + 'DeepJet').to_numpy() for jet in jets])).sum(axis=1)/6
         except: self.btag_avg = np.column_stack(([self.get(jet + 'btag').to_numpy() for jet in jets])).sum(axis=1)/6

         self.btag_avg2 = np.column_stack(([self.get(jet + 'btag').to_numpy()**2 for jet in jets])).sum(axis=1)/6

         self.initialize_vars()
        
   def norm(self, nevents):
      return int(nevents*self.scale)

   def keys(self):
      return self.tree.keys()

   def find_key(self, start):
      for key in self.keys():
         if start in key : print(key)

   def get(self, key, library='ak'):
      """Returns the key.
      Use case: Sometimes keys are accessed directly, e.g. tree.key, but other times, the key may be accessed from a list of keys, e.g. tree.get(['key']). This function handles the latter case.
      """
      if key in self.tree.keys():
         return self.tree[key].array(library=library)
      else:
         arr = getattr(self, key)
         if library=='np' and not isinstance(arr, np.ndarray): arr = arr.to_numpy()
         return arr
    
   def np(self, key):
      """Returns the key as a numpy array."""
      np_arr = self.get(key, library='np')
      if not isinstance(np_arr, np.ndarray): np_arr = np_arr.to_numpy()
      return np_arr

   def initialize_vars(self):
      """Initialize variables that don't exist in the original ROOT tree."""
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

      # HX = HX_b1 + HX_b2
      # H1 = H1_b1 + H1_b2
      # H2 = H2_b1 + H2_b2

      # Y = H1 + H2

      self.HX_dr = HX_b1.deltaR(HX_b2)
      self.H1_dr = H1_b1.deltaR(H1_b2)
      self.H2_dr = H2_b1.deltaR(H2_b2)

      self.HX_H1_dEta = HX.deltaEta(H1)
      self.H1_H2_dEta = H1.deltaEta(H2)
      self.H2_HX_dEta = H2.deltaEta(HX)

      self.HX_H1_dPhi = HX.deltaPhi(H1)
      self.H1_H2_dPhi = H1.deltaPhi(H2)
      self.H2_HX_dPhi = H2.deltaPhi(HX)

      self.HX_H1_dr = HX.deltaR(H1)
      self.HX_H2_dr = H2.deltaR(HX)
      self.H1_H2_dr = H1.deltaR(H2)
      
      self.Y_HX_dr = Y.deltaR(HX)

      self.HX_costheta = np.cos(HX.P4.theta)
      self.H1_costheta = np.cos(H1.P4.theta)
      self.H2_costheta = np.cos(H2.P4.theta)

      self.HX_H1_dr = HX.deltaR(H1)
      self.H1_H2_dr = H2.deltaR(H1)
      self.H2_HX_dr = HX.deltaR(H2)

      # X = HX + H1 + H2
      self.X_m = X.m

   def hist(self, var, bins, label=None, ax=None, title=True, lines=None, **kwargs):
   
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
            n = Hist(arr, bins=bins, label=lbl, ax=ax, weights=self.scale, color=c, linestyle=s)
            n_all.append(n)
            if n.max() > n_max: n_max = n.max()
         ax.legend()
         return n_all
      elif isinstance(var, str):
         var = self.get(var)
      
      if label is not None:
         n = Hist(var, bins=bins, ax=ax, weights=self.scale, label=label)
      else:
         n = Hist(var, bins=bins, ax=ax, weights=self.scale)
      n_max = n.max()

      if isinstance(lines, list):
         for val in lines:
            ax.plot([val,val],[0, n_max], 'gray', '--', alpha=0.5)
      # print(np.around(n))
      
      if title:
         ax.set_title(self.sample)

      if np.issubdtype(njet_bins.dtype, int):
         ax.tick_params(axis='x', which='minor', color='white')
         ax.xaxis.set_ticks(bins)
         # ax.xaxis.set_tick_params(which='minor', bottom=False)
      
      ax.set_ylabel('AU')

      return n


   

   # def rectangular_region(self, config):
   #    """Defines rectangular region masks."""
   #    SR_edge = float(config['rectangular']['maxSR'])
   #    VR_edge = float(config['rectangular']['maxVR'])
   #    CR_edge = float(config['rectangular']['maxCR'])
   #    if CR_edge == -1: CR_edge = 9999

   #    score_cut = float(config['score']['threshold'])

   #    self.SR_edge = SR_edge
   #    self.VR_edge = VR_edge
   #    self.CR_edge = CR_edge

   #    higgs = ['HX_m', 'H1_m', 'H2_m']
   #    Dm_cand = np.column_stack(([abs(self.get(mH,'np') - 125) for mH in higgs]))
   #    SR_mask = ak.all(Dm_cand <= SR_edge, axis=1) # SR
   #    VR_mask = ak.all(Dm_cand > SR_edge, axis=1) & ak.all(Dm_cand <= VR_edge, axis=1) # VR
   #    CR_mask = ak.all(Dm_cand > VR_edge, axis=1) & ak.all(Dm_cand <= CR_edge, axis=1) # CR

   #    ls_mask = self.btag_avg < score_cut # ls
   #    hs_mask = self.btag_avg >= score_cut # hs

   #    self.V_CRls_mask = CR_mask & ls_mask
   #    self.V_CRhs_mask = CR_mask & hs_mask
   #    self.V_SRls_mask = VR_mask & ls_mask
   #    self.V_SRhs_mask = VR_mask & hs_mask
   #    self.A_CRls_mask = np.zeros_like(CR_mask)
   #    self.A_CRhs_mask = np.zeros_like(CR_mask)
   #    self.A_SRls_mask = SR_mask & ls_mask
   #    self.A_SRhs_mask = SR_mask & hs_mask

   #    if self._isdata:
   #       print(f"   SR_edge   {int(SR_edge)}")
   #       print(f"   VR_edge   {int(VR_edge)}")
   #       print(f"   CR_edge   {int(CR_edge)}")
   #       print(" -----------------")
   #       print()
         
   #       # blinded_mask = VR_mask | CR_mask
   #       self.dat_mX_V_CRls = self.np('X_m')[self.V_CRls_mask]
   #       self.dat_mX_V_CRhs = self.np('X_m')[self.V_CRhs_mask]
   #       self.dat_mX_V_SRls = self.np('X_m')[self.V_SRls_mask]
   #       self.dat_mX_V_SRhs = self.np('X_m')[self.V_SRhs_mask]
   #       self.dat_mX_A_CRls = np.zeros_like(self.np('X_m'))
   #       self.dat_mX_A_CRhs = np.zeros_like(self.np('X_m'))
   #       self.dat_mX_A_SRls = self.np('X_m')[self.A_SRls_mask]
   #       # self.dat_mX_blinded = self.np('X_m')[blinded_mask]
   #    else:
   #       return self.np('X_m')[self.SRhs_mask]

   #    self._rect_region_bool = True

   def spherical_region(self, config):
      if isinstance(config, str):
         cfg = config
         from configparser import ConfigParser
         config = ConfigParser()
         config.optionxform = str
         config.read(cfg)

      """Defines spherical estimation region masks."""
      AR_center = float(config['spherical']['ARcenter'])
      SR_edge   = float(config['spherical']['rInner'])
      CR_edge   = float(config['spherical']['rOuter'])

      VR_center = int(AR_center + (SR_edge + CR_edge)/np.sqrt(2))

      higgs = ['HX_m', 'H1_m', 'H2_m']

      Dm_cand = np.column_stack(([abs(self.get(mH,'np') - AR_center) for mH in higgs]))
      Dm_cand = Dm_cand * Dm_cand
      Dm_cand = Dm_cand.sum(axis=1)
      Dm_cand = np.sqrt(Dm_cand)
      self.A_SR_mask = Dm_cand <= SR_edge # Analysis SR
      self.A_CR_mask = (Dm_cand > SR_edge) & (Dm_cand <= CR_edge) # Analysis CR

      Dm_cand = np.column_stack(([abs(self.get(mH,'np') - VR_center) for mH in higgs]))
      Dm_cand = Dm_cand * Dm_cand
      Dm_cand = Dm_cand.sum(axis=1)
      Dm_cand = np.sqrt(Dm_cand)
      self.V_SR_mask = Dm_cand <= SR_edge # Validation SR
      self.V_CR_mask = (Dm_cand > SR_edge) & (Dm_cand <= CR_edge) # Validation CR

      score_cut = float(config['score']['threshold'])
      ls_mask = self.btag_avg < score_cut # ls
      hs_mask = self.btag_avg >= score_cut # hs


      # b_cut = float(config['score']['n'])
      # self.nloose_b = ak.sum(self.get('jet_btag') > 0.0490, axis=1)
      # self.nmedium_b = ak.sum(self.get('jet_btag') > 0.2783, axis=1)
      # ls_mask = self.nmedium_b < b_cut # ls
      # hs_mask = self.nmedium_b >= b_cut # hs

      self.A_CRls_mask = self.A_CR_mask & ls_mask
      self.A_CRhs_mask = self.A_CR_mask & hs_mask
      self.A_SRls_mask = self.A_SR_mask & ls_mask
      self.A_SRhs_mask = self.A_SR_mask & hs_mask

      self.V_CRls_mask = self.V_CR_mask & ls_mask
      self.V_CRhs_mask = self.V_CR_mask & hs_mask
      self.V_SRls_mask = self.V_SR_mask & ls_mask
      self.V_SRhs_mask = self.V_SR_mask & hs_mask

      self.hs_mask = hs_mask
      self.ls_mask = ls_mask

      if self._isdata:
         print(f"VR_center   = {VR_center}")
         print(f"SR_edge     = {int(SR_edge)}")
         print(f"CR_edge     = {int(CR_edge)}")
         print( "--------------------")
         # print(f"b tag score = {score_cut}")
         # print(f"b tag score = {score_cut}")
         # print( "--------------------")
         print()

         self.dat_mX_V_CRhs = self.np('X_m')[self.V_CRhs_mask]
         self.dat_mX_V_CRls = self.np('X_m')[self.V_CRls_mask]
         self.dat_mX_V_SRhs = self.np('X_m')[self.V_SRhs_mask]
         self.dat_mX_V_SRls = self.np('X_m')[self.V_SRls_mask]
         self.dat_mX_A_CRhs = self.np('X_m')[self.A_CRhs_mask]
         self.dat_mX_A_CRls = self.np('X_m')[self.A_CRls_mask]
         self.dat_mX_A_SRls = self.np('X_m')[self.A_SRls_mask]
      
      self._sphere_region_bool = True

   def region_yield(self, definition='rect', normalized=False, config=None, SR_hs_est=None, circ_region=None):
      """Prints original or normalized estimation region yields."""

      if definition == 'rect':
         if not self._rect_region_bool: self.rectangular_region(config)

         CRls_yield = int(ak.sum(self.CRls_mask)*self.scale)
         CRhs_yield = int(ak.sum(self.CRhs_mask)*self.scale)
         VRls_yield = int(ak.sum(self.VRls_mask)*self.scale)
         VRhs_yield = int(ak.sum(self.VRhs_mask)*self.scale)
         SRls_yield = int(ak.sum(self.SRls_mask)*self.scale)
         SRhs_yield = int(ak.sum(self.SRhs_mask)*self.scale)

         if self._isdata:
               SRhs_yield = 0
         if SR_hs_est is not None:
               SRhs_yield = int(SR_hs_est)

         if normalized:
               total = CRls_yield + CRhs_yield + VRls_yield + VRhs_yield + SRls_yield + SRhs_yield
               CRls_yield = int((CRls_yield / total)*100)
               CRhs_yield = int((CRhs_yield / total)*100)
               VRls_yield = int((VRls_yield / total)*100)
               VRhs_yield = int((VRhs_yield / total)*100)
               SRls_yield = int((SRls_yield / total)*100)
               SRhs_yield = int((SRhs_yield / total)*100)
         
         yields = [CRls_yield, CRhs_yield, VRls_yield, VRhs_yield, SRls_yield, SRhs_yield]
         cols = ['CR low-avg', 'CR high-avg', 'VR low-avg', 'VR high-avg', 'SR low-avg', 'SR high-avg']
         data = {col:[value] for col,value in zip(cols,yields)}

         df = pd.DataFrame(data=data, columns=cols)
         pd.set_option('display.colheader_justify', 'center')
         print(df.to_string(index=False))
      elif definition == 'sphere':
         if not self._sphere_region_bool: self.spherical_region(config)

         A_CRls_yield = int(ak.sum(self.A_CRls_mask)*self.scale)
         A_CRhs_yield = int(ak.sum(self.A_CRhs_mask)*self.scale)
         A_SRls_yield = int(ak.sum(self.A_SRls_mask)*self.scale)
         A_SRhs_yield = int(ak.sum(self.A_SRhs_mask)*self.scale)

         V_CRls_yield = int(ak.sum(self.V_CRls_mask)*self.scale)
         V_CRhs_yield = int(ak.sum(self.V_CRhs_mask)*self.scale)
         V_SRls_yield = int(ak.sum(self.V_SRls_mask)*self.scale)
         V_SRhs_yield = int(ak.sum(self.V_SRhs_mask)*self.scale)

         if self._isdata:
               A_SRhs_yield = 0
         pred = False
         if SR_hs_est is not None:
               pred = True
               if circ_region == 'AR':
                  A_SRhs_yield = SR_hs_est
               elif circ_region == 'VR':
                  V_SRhs_yield = SR_hs_est

         A_total = A_CRls_yield + A_CRhs_yield + A_SRls_yield + A_SRhs_yield
         V_total = V_CRls_yield + V_CRhs_yield + V_SRls_yield + V_SRhs_yield

         if normalized:
               A_CRls_yield = A_CRls_yield / A_total
               A_CRhs_yield = A_CRhs_yield / A_total
               A_SRls_yield = A_SRls_yield / A_total
               A_SRhs_yield = A_SRhs_yield / A_total

               V_CRls_yield = V_CRls_yield / V_total
               V_CRhs_yield = V_CRhs_yield / V_total
               V_SRls_yield = V_SRls_yield / V_total
               V_SRhs_yield = V_SRhs_yield / V_total

         regions = np.repeat(['Analysis Region', 'Validation Region'],4)
         cols = np.tile(['CR low-avg', 'CR high-avg', 'SR low-avg', 'SR high-avg'],2)
         if pred: cols[3], cols[7] = 'SR high-avg (predicted)', 'SR high-avg (predicted)'
         yields = [A_CRls_yield, A_CRhs_yield, A_SRls_yield, A_SRhs_yield, V_CRls_yield, V_CRhs_yield, V_SRls_yield, V_SRhs_yield]
         yields = [int(value) for value in yields]

         index = [(region, col) for region,col in zip(regions,cols)]

         index = pd.MultiIndex.from_tuples(index)

         yields = pd.Series(yields, index=index)
         print("\nPrinting yields.")
         print(yields)

   def get_df(self, mask, variables):
      features = {}
      for var in variables:
         features[var] = abs(self.get(var)[mask])
      df = pd.DataFrame(features)
      return df

   def train_bdt(self, config, ls_mask, hs_mask):

      Nestimators  = int(config['BDT']['Nestimators'])
      learningRate = float(config['BDT']['learningRate'])
      maxDepth     = int(config['BDT']['maxDepth'])
      minLeaves    = int(config['BDT']['minLeaves'])
      GBsubsample  = float(config['BDT']['GBsubsample'])
      randomState  = int(config['BDT']['randomState'])
      variables = config['BDT']['variables']
      if isinstance(variables, str):
         variables = variables.split(", ")
      self.variables = variables

      self.TF = sum(hs_mask)/sum(ls_mask)
      ls_weights = np.ones(ak.sum(ls_mask))*self.TF
      hs_weights = np.ones(ak.sum([hs_mask]))

      df_ls = self.get_df(ls_mask, variables)
      df_hs = self.get_df(hs_mask, variables)

      np.random.seed(randomState) #Fix any random seed using numpy arrays
      print(".. calling reweight.GBReweighter")
      reweighter_base = reweight.GBReweighter(
         n_estimators=Nestimators, 
         learning_rate=learningRate, 
         max_depth=maxDepth, 
         min_samples_leaf=minLeaves,
         gb_args={'subsample': GBsubsample})

      print(".. calling reweight.FoldingReweighter")
      reweighter = reweight.FoldingReweighter(reweighter_base, random_state=randomState, n_folds=2, verbose=False)

      print(".. calling reweighter.fit")
      reweighter.fit(df_ls,df_hs,ls_weights,hs_weights)
      self.reweighter = reweighter

   def bdt_prediction(self, ls_mask):
   
      df_ls = self.get_df(ls_mask, self.variables)
      initial_weights = np.ones(ak.sum(ls_mask))*self.TF

      weights_pred = self.reweighter.predict_weights(df_ls,initial_weights,lambda x: np.mean(x, axis=0))

      return weights_pred

   def bdt_process(self, scheme, config):
      if scheme == 'rect':
         if not self._rect_region_bool: self.rectangular_region(config)

         print(".. training BDT in CR")
         self.train_bdt(config, self.CRls_mask, self.V_CRhs_mask)
         print(".. predicting weights in CR")
         self.V_CR_weights = self.bdt_prediction(self.V_CRls_mask)
         print(".. performing kstest")
         self.ks_test(self.CRls_mask, self.V_CRhs_mask, BDT=False)

         print(".. predicting weights in VR")
         self.V_SR_weights = self.bdt_prediction(self.V_SRls_mask)

         print(".. predicting weights in SR")
         self.A_SR_weights = self.bdt_prediction(self.A_SRls_mask)

         CRls = self.CRls_mask
         CRhs = self.CRhs_mask
         VRls = self.VRls_mask
         VRhs = self.VRhs_mask
         SRls = self.SRls_mask

      elif scheme == 'sphere':
         if not self._sphere_region_bool: self.spherical_region(config)

         print(".. training BDT in V_CR")
         self.train_bdt(config, self.V_CRls_mask, self.V_CRhs_mask)
         print(".. predicting weights in CR")
         self.V_CR_weights = self.bdt_prediction(self.V_CRls_mask)
         print()
         # print(".. performing kstest")
         # self.V_CR_kstest = self.ks_test(self.V_CRls_mask, self.V_CRhs_mask, self.V_CR_weights)
         # self.V_CR_prob_w, self.V_CR_prob_unw = self.get_prob(self.V_CRls_mask, self.V_CRhs_mask, self.V_CR_weights)
               
         print(".. predicting weights in V_SR")
         self.V_SR_weights = self.bdt_prediction(self.V_SRls_mask)
         print()
         # self.V_SR_kstest_pre = self.ks_test(self.V_SRls_mask, self.V_SRhs_mask, np.ones_like(self.V_SR_weights))
         # self.V_SR_kstest = self.ks_test(self.V_SRls_mask, self.V_SRhs_mask, self.V_SR_weights)
         # self.V_SR_prob_w, self.V_SR_prob_unw = self.get_prob(self.V_SRls_mask, self.V_SRhs_mask, self.V_SR_weights)

         print(".. training BDT in A_CR")
         self.train_bdt(config, self.A_CRls_mask, self.A_CRhs_mask)
         self.A_CR_weights = self.bdt_prediction(self.A_CRls_mask)
         print()
         # self.A_CR_kstest = self.ks_test(self.A_CRls_mask, self.A_CRhs_mask, self.A_CR_weights)
         # self.A_CR_prob_w, self.A_CR_prob_unw = self.get_prob(self.A_CRls_mask, self.A_CRhs_mask, self.A_CR_weights)

         print(".. predicting weights in A_SR\n")
         self.A_SR_weights = self.bdt_prediction(self.A_SRls_mask)

         CRls = self.V_CRls_mask
         CRhs = self.V_CRhs_mask
         VRls = self.V_SRls_mask
         VRhs = self.V_SRhs_mask
         SRls = self.A_SRls_mask

      self.get_stats(CRls, CRhs, VRls, VRhs, SRls)

   def get_stats(self, CRls, CRhs, VRls, VRhs, SRls):
      N_CRls = ak.sum(CRls)
      N_CRhs = ak.sum(CRhs)
      N_VRhs = ak.sum(VRhs)
      N_VRpred = ak.sum(self.V_SR_weights)
      N_SRpred = ak.sum(self.A_SR_weights)

      s_CRls = 1/np.sqrt(N_CRls)
      s_CRhs = 1/np.sqrt(N_CRhs)
      s_VRpred = np.sqrt(ak.sum(self.V_SR_weights**2))
      s_SRpred = np.sqrt(ak.sum(self.A_SR_weights**2))

      
      self.bkg_vrpred = round(1 + np.where(s_VRpred/N_VRpred > s_SRpred/N_SRpred, np.sqrt((s_VRpred/N_VRpred)**2 - (s_SRpred/N_SRpred)**2), 0), 2)
      self.bkg_crtf = round(1 + np.sqrt((s_CRls/N_CRls)**2 + (s_CRhs/N_CRhs)**2), 2)
      err = np.sqrt(N_VRpred + s_VRpred**2 + (N_VRpred*(self.bkg_crtf-1))**2)
      k = np.sqrt((N_VRhs - N_VRpred)**2 - err**2)
      if np.isnan(k): k = 0
      self.bkg_vr_normval = round(1 + k/N_VRpred, 2)

      s_obs = 1/N_VRhs
      s_mod = 1/N_VRpred

      esum1 = N_VRhs**2 / s_obs
      esum2 = N_VRpred**2 / s_VRpred**2
      
      n_obs,e = np.histogram(self.dat_mX_V_SRhs, bins=np.linspace(375,2000,40))
      n_mod,e = np.histogram(self.dat_mX_V_SRls, weights=self.V_SR_weights, bins=np.linspace(375,2000,40))
      rsum1 = s_obs * n_obs
      rsum2 = s_mod * n_mod

      dfmax = np.max(np.abs(rsum1-rsum2))
      z = dfmax*np.sqrt(esum1*esum2/(esum1+esum2))

      p = 0
      for j in range(1,1000):
         p += 2*(-1)**(j-1)*np.exp(-2*j**2*z**2)
      
      # ks =  ks_2samp_weighted(self.dat_mX_V_SRls, self.dat_mX_V_SRhs, weights1=self.V_SR_weights, weights2=np.ones_like(self.dat_mX_V_SRhs))
      
   #   print("CALCULATED PROBABILITY",p)
      # print("V_SR KS test max",round(ks,3))

   def ks_test(self, ls_mask, hs_mask, weights, BDT=True):
      ksresults = [] 
      original = self.get_df(ls_mask, self.variables)
      target = self.get_df(hs_mask, self.variables)

      if not BDT: weights1 = weights1 = np.ones(len(original))*self.TF
      else: weights1 = weights

      weights2 = np.ones(len(target), dtype=int)

      cols = []
      skip = ['HX_eta', 'H1_eta', 'H2_eta', 'HX_H1_dEta', 'H1_H2_dEta', 'H2_HX_dEta']
      for i, column in enumerate(self.variables):
         # if column in skip: continue
         cols.append(column)
         ks =  ks_2samp_weighted(original[column], target[column], weights1=weights1, weights2=weights2)
         ksresults.append(ks)
         # print(i)
   #   additional = [self.np('X_m')]
   #   for extra in additional:
   #       original = extra[ls_mask]
   #       target = extra[hs_mask]
   #       ks =  ks_2samp_weighted(original, target, weights1=weights1, weights2=weights2)
   #       ksresults.append(ks)
   #   cols.append('MX')
      ksresults = np.asarray(ksresults)
      self.ksmax = round(ksresults.max(),3)
      
      if self.ksmax < 0.01: 
         self.kstest = True
         print(f"[{Fore.GREEN}SUCCESS{Fore.RESET}] ks-val = {self.ksmax}")
      else: 
         self.kstest = False
         print(f"[{Fore.RED}FAILURE{Fore.RESET}] ks-val = {self.ksmax}")

      return ksresults

   def get_prob(self, ls_mask, hs_mask, weights):

      p_weighted, p_unweighted = [], []
      for var in self.variables:
         target = abs(self.get(var, 'np'))[hs_mask]
         original = abs(self.get(var, 'np'))[ls_mask]

         if min(target) < min(original): b_min = min(target)
         else: b_min = min(original)
         if max(target) > max(original): b_max = max(target)
         else: b_max = max(original)
         if b_max >= 1000: b_max = b_max / 2

         bins = np.linspace(b_min,b_max,20)

         n_target    , _ = np.histogram(target, bins=bins)
         n_unweighted, _ = np.histogram(original, bins=bins)
         n_weighted  , _ = np.histogram(original, bins=bins, weights=weights)

         try: del h_target, h_weighted, h_unweighted
         except: pass

         h_target = ROOT.TH1D(var+"1",var+"1",len(n_target),array('d',list(bins)))
         h_weighted = ROOT.TH1D(var+"2",var+"2",len(n_weighted),array('d',list(bins)))
         h_unweighted = ROOT.TH1D(var+"3",var+"3",len(n_unweighted),array('d',list(bins)))
         for i,(n_t, n_w, n_un) in enumerate(zip(n_target, n_weighted, n_unweighted)):
            h_target.SetBinContent(i+1, n_t)
            h_weighted.SetBinContent(i+1, n_w)
            h_unweighted.SetBinContent(i+1, n_un)

         p_w = h_target.KolmogorovTest(h_weighted)
         p_unw = h_target.KolmogorovTest(h_unweighted)

         p_weighted.append(p_w)
         p_unweighted.append(p_unw)

      return p_weighted, p_unweighted


def presels(tree, pt=20, eta=2.5, jetid=6, puid=6):
   pt_mask = ak.sum(tree.jet_pt > pt, axis=1) >= 6
   eta_mask = ak.sum(abs(tree.jet_eta) < eta, axis=1) >= 6
   jetid_mask = ak.sum(tree.jet_id == jetid, axis=1) >= 6
   pt_under50 = tree.jet_pt < 50
   puid_mask = ak.sum(tree.jet_puid[pt_under50] == puid, axis=1) + ak.sum(~pt_under50, axis=1) >= 6

   presel_mask = pt_mask & eta_mask & jetid_mask & puid_mask
   return tree[presel_mask]



class DataTraining():
   def __init__(self, config, b_cut=0.55, gnn_cut=0.76):
      data_tree_ar = uproot.open("/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixB/analysis/sixBanalysis/data_ar.root:sixBtree")
      data_tree_vr = uproot.open("/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixB/analysis/sixBanalysis/data_vr.root:sixBtree")

      Nestimators  = int(config['BDT']['Nestimators'])
      learningRate = float(config['BDT']['learningRate'])
      maxDepth     = int(config['BDT']['maxDepth'])
      minLeaves    = int(config['BDT']['minLeaves'])
      GBsubsample  = float(config['BDT']['GBsubsample'])
      randomState  = int(config['BDT']['randomState'])
      variables = config['BDT']['variables']
      if isinstance(variables, str):
         variables = variables.split(", ")

      data_gnn_score_ar = ak.sort(data_tree_ar['jet_gnn_score'].array(), ascending=False, axis=1)[:,:6]
      data_gnn_score_vr = ak.sort(data_tree_vr['jet_gnn_score'].array(), ascending=False, axis=1)[:,:6]

      data_gnn_avg_ar = ak.sum(data_gnn_score_ar, axis=1) / 6
      data_gnn_avg_vr = ak.sum(data_gnn_score_vr, axis=1) / 6

      asr_gnn_mask = data_gnn_avg_ar > 0.75
      asr_mask = data_tree_ar['bkg_mask'].array() # already accounts for cut on average btag score
      acr_mask = ~asr_mask & asr_gnn_mask
      asr_mask = asr_mask & asr_gnn_mask

      # analysis region
      data_avg_btag_ar = data_tree_ar['jet_avg_btag'].array()
      asr_avg_b = data_avg_btag_ar[asr_mask]
      acr_avg_b = data_avg_btag_ar[acr_mask]

      acr_ls_mask = (acr_avg_b < b_cut)
      acr_hs_mask = (acr_avg_b > b_cut)

      # validation region
      data_avg_btag_vr = data_tree_vr['jet_avg_btag'].array()

      vr_gnn_mask = data_gnn_avg_vr > 0.75
      vsr_mask = data_tree_vr['v_sr_mask'].array() & vr_gnn_mask
      vsr_avg_b = data_avg_btag_vr[vsr_mask]
      vsr_ls_mask = (vsr_avg_b < b_cut)
      vsr_hs_mask = (vsr_avg_b > b_cut)

      vcr_mask = data_tree_vr['v_cr_mask'].array() & vr_gnn_mask
      vcr_avg_b = data_avg_btag_vr[vcr_mask]
      vcr_ls_mask = (vcr_avg_b < b_cut)
      vcr_hs_mask = (vcr_avg_b > b_cut)

      asr_X_m = data_tree_ar['X_m'].array()[asr_mask]
      acr_X_m = data_tree_ar['X_m'].array()[acr_mask]
      vsr_X_m = data_tree_vr['X_m'].array()[vsr_mask]
      vcr_X_m = data_tree_vr['X_m'].array()[vcr_mask]

      asr_ls_X_m = data_tree_ar['X_m'].array()[asr_mask] # low score by construction

      acr_ls_X_m = data_tree_ar['X_m'].array()[acr_mask][acr_ls_mask]
      vsr_ls_X_m = data_tree_vr['X_m'].array()[vsr_mask][vsr_ls_mask]
      vcr_ls_X_m = data_tree_vr['X_m'].array()[vcr_mask][vcr_ls_mask]

      acr_hs_X_m = data_tree_ar['X_m'].array()[acr_mask][acr_hs_mask]
      vsr_hs_X_m = data_tree_vr['X_m'].array()[vsr_mask][vsr_hs_mask]
      vcr_hs_X_m = data_tree_vr['X_m'].array()[vcr_mask][vcr_hs_mask]

      acr_TF = sum(acr_hs_mask)/sum(acr_ls_mask)
      vcr_TF = sum(vcr_hs_mask)/sum(vcr_ls_mask)

      acr_ls_weights = np.ones(ak.sum(acr_ls_mask))*acr_TF
      acr_hs_weights = np.ones(ak.sum([acr_hs_mask]))
      vcr_ls_weights = np.ones(ak.sum(vcr_ls_mask))*vcr_TF
      vcr_hs_weights = np.ones(ak.sum(vcr_hs_mask))

      acr_ls_features = {}
      acr_hs_features = {}
      asr_ls_features = {}
      vcr_ls_features = {}
      vcr_hs_features = {}
      vsr_ls_features = {}
      vsr_hs_features = {}
      for var in variables:
         acr_ls_features[var] = abs(data_tree_ar[var].array())[acr_ls_mask]
         acr_hs_features[var] = abs(data_tree_ar[var].array())[acr_hs_mask]
         asr_ls_features[var] = abs(data_tree_ar[var].array())[asr_mask]
         vcr_ls_features[var] = abs(data_tree_vr[var].array()[vcr_ls_mask])
         vcr_hs_features[var] = abs(data_tree_vr[var].array()[vcr_hs_mask])
         vsr_ls_features[var] = abs(data_tree_vr[var].array())[vsr_ls_mask]
         vsr_hs_features[var] = abs(data_tree_vr[var].array())[vsr_hs_mask]
      acr_ls_df = pd.DataFrame(acr_ls_features)
      acr_hs_df = pd.DataFrame(acr_hs_features)
      asr_ls_df = pd.DataFrame(asr_ls_features)
      vcr_ls_df = pd.DataFrame(vcr_ls_features)
      vcr_hs_df = pd.DataFrame(vcr_hs_features)
      vsr_ls_df = pd.DataFrame(vsr_ls_features)
      vsr_hs_df = pd.DataFrame(vsr_hs_features)

      acr_reweighter_base = reweight.GBReweighter(
         n_estimators=Nestimators, 
         learning_rate=learningRate, 
         max_depth=maxDepth, 
         min_samples_leaf=minLeaves,
         gb_args={'subsample': GBsubsample})

      acr_reweighter = reweight.FoldingReweighter(acr_reweighter_base, random_state=randomState, n_folds=2, verbose=False)
      acr_reweighter.fit(acr_ls_df,acr_hs_df,acr_ls_weights,acr_hs_weights)

      asr_initial_weights = np.ones(ak.sum(asr_mask))*acr_TF
      acr_initial_weights = np.ones(ak.sum(acr_ls_mask))*acr_TF
      vsr_initial_weights = np.ones(ak.sum(vsr_ls_mask))*vcr_TF

      asr_weights_pred = acr_reweighter.predict_weights(asr_ls_df,asr_initial_weights,lambda x: np.mean(x, axis=0))
      acr_weights_pred = acr_reweighter.predict_weights(acr_ls_df,acr_initial_weights,lambda x: np.mean(x, axis=0))

      n_acr_hs, _ = np.histogram(
         acr_hs_X_m.to_numpy(), 
         bins=m_bins
         )
      
      n_acr_hs_pred, _ = np.histogram(
         acr_ls_X_m.to_numpy(),
         bins=m_bins,
         weights=acr_weights_pred
      )

      acr_TF_bvb = n_acr_hs / n_acr_hs_pred

      n_asr_hs_pred, _ = np.histogram(
         asr_X_m.to_numpy(),
         bins=m_bins,
         weights=asr_weights_pred
      )

      n_asr_hs_pred, _ = np.histogram(
         x_X_m,
         bins=m_bins,
         weights=n_asr_hs_pred*acr_TF_bvb
      )

      ROOTHist(n_asr_hs_pred, 'data', 'data')

      vcr_reweighter_base = reweight.GBReweighter(
         n_estimators=Nestimators, 
         learning_rate=learningRate, 
         max_depth=maxDepth, 
         min_samples_leaf=minLeaves,
         gb_args={'subsample': GBsubsample})

      vcr_reweighter = reweight.FoldingReweighter(vcr_reweighter_base, random_state=randomState, n_folds=2, verbose=False)
      vcr_reweighter.fit(vcr_ls_df,vcr_hs_df,vcr_ls_weights,vcr_hs_weights)

      vsr_weights_pred = vcr_reweighter.predict_weights(vsr_ls_df,vsr_initial_weights,lambda x: np.mean(x, axis=0))
      vcr_weights_pred = vcr_reweighter.predict_weights(vcr_ls_df,np.ones(ak.sum(vcr_ls_mask))*vcr_TF,lambda x: np.mean(x, axis=0))

      n_vcrls, _ = np.histogram(
         vcr_ls_X_m.to_numpy(),
         bins=m_bins
      )

      n_vcrhs, _ = np.histogram(
         vcr_hs_X_m.to_numpy(),
         bins=m_bins
      )

      n_vcrhs_pred, _ = np.histogram(
         vcr_ls_X_m.to_numpy(),
         bins=m_bins,
         weights=vcr_weights_pred
      )
      vcr_TF_bvb = np.nan_to_num(n_vcrhs / n_vcrhs_pred, 0)

      n_vsrhs, _ = np.histogram(
         vsr_hs_X_m.to_numpy(),
         bins=m_bins
      )

      n_vsrhs_pred, _ = np.histogram(
         vsr_ls_X_m.to_numpy(),
         bins=m_bins,
         weights=vsr_weights_pred
      )

      n_vsrhs_pred, _ = np.histogram(
         x_X_m,
         bins=m_bins,
         weights=n_vsrhs_pred*vcr_TF_bvb
      )

      s_asrhs_pred, _ = np.histogram(
         asr_ls_X_m.to_numpy(),
         bins=m_bins,
         weights=asr_weights_pred**2
      )

      s_asrhs_pred, _ = np.histogram(
         x_X_m,
         bins=m_bins,
         weights=s_asrhs_pred*vcr_TF_bvb
      )

      s_vsrhs_pred, _ = np.histogram(
         vsr_ls_X_m.to_numpy(),
         bins=m_bins,
         weights=vsr_weights_pred**2
      )

      s_vsrhs_pred, _ = np.histogram(
         x_X_m,
         bins=m_bins,
         weights=s_vsrhs_pred*vcr_TF_bvb
      )


      n_vcr_ls = n_vcrls.sum()
      n_vcr_hs = n_vcrhs.sum()

      n_vsr_pred = n_vsrhs_pred.sum()
      n_asr_pred = n_asr_hs_pred.sum()

      s_vcr_ls = 1 / np.sqrt(n_vcr_ls)
      s_vcr_hs = 1 / np.sqrt(n_vcr_hs)
      s_vsr_pred = 1 / np.sqrt(ak.sum(s_vsrhs_pred))
      s_asr_pred = 1 / np.sqrt(ak.sum(s_asrhs_pred))

      self.bkg_vrpred = round(1 + np.where(s_vsr_pred/n_vsr_pred > s_asr_pred/n_asr_pred, np.sqrt((s_vsr_pred/n_vsr_pred)**2 - (s_asr_pred/n_asr_pred)**2), 0), 2)

      self.cr_tf = 1 + np.sqrt(1/n_vcr_hs + 1/n_vcr_ls)

      self.vr_prec = 1 + np.sqrt(1/n_vsr_pred*100**2 - 1/n_asr_pred*100**2)





class GNNSelection():
   _var_keys = ['n_jet', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_m', 'jet_btag', 'jet_avg_btag', 'asr_mask', 'acr_mask', 'X_m', 'jet_signalId', 'jet_gnn_score']

   _eos_base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/NMSSM_presel'

   _root_base = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixB/analysis/sixBanalysis/gnn'

   def __init__(self, filename, year=2018, b_cut=0.55, gnn_cut=0.76):

      MX_ind = re.search('MX_', filename)
      MY_ind = re.search('MY_', filename)

      self.mass_name = filename[MX_ind.start():]

      MX = int(filename[MX_ind.end():MY_ind.start()-1])
      MY = int(filename[MY_ind.end():])

      eos_file = f"{self._eos_base}/{filename}/ntuple.root"
      gnn_file = f"{self._root_base}/{self.mass_name}.root"

      tree = uproot.open(f"{gnn_file}:sixBtree")
      cutflow = uproot.open(f"{eos_file}:h_cutflow")
      cutflow = cutflow.to_numpy()[0]
      total = cutflow[0]
      xsec = 0.3 # pb

      nevents = len(tree['n_jet'].array())
      cutflow = np.append(cutflow, nevents)
      self.scale = lumiMap[2018][0] * xsec / total
      self.cutflow = round(cutflow * self.scale)

      self.tree = tree

      for k in self._var_keys:
            setattr(self, k, tree[k].array())

      self.jet_gnn_score = ak.sort(tree['jet_gnn_score'].array(), ascending=False, axis=1)[:,:6]
      self.avg_gnn = ak.sum(self.jet_gnn_score, axis=1) / 6

      self.gnn_mask = self.avg_gnn > gnn_cut
      self.btag_mask = self.jet_avg_btag > b_cut
      self.asr_mask = self.asr_mask & self.gnn_mask & self.btag_mask

      n_sr_hs, _ = np.histogram(
         self.X_m[self.asr_mask],
         bins=m_bins,
         weights=np.ones_like(self.X_m[self.asr_mask])*self.scale
      )

      ROOTHist(n_sr_hs, self.mass_name, 'signal')

      self.get_limits()

   def keys(self):
      return self.tree.keys()

   def get_limits(self):
      import pyhf
      pyhf.set_backend("jax")
      h_bkg = uproot.open("ml/gnn/data.root:data")
      h_sig = uproot.open(f"ml/gnn/{self.mass_name}.root:signal")

      norm = 2*np.sqrt(np.sum(h_bkg.errors()**2))/h_sig.counts().sum()
      w = pyhf.simplemodels.uncorrelated_background(
         signal=(norm*h_sig.counts()).tolist(), bkg=h_bkg.counts().tolist(), bkg_uncertainty=h_bkg.errors().tolist()
      )
      data = h_bkg.counts().tolist()+w.config.auxdata

      poi = np.linspace(0,5,11)
      level = 0.05

      obs_limit, exp_limit = pyhf.infer.intervals.upperlimit(
               data, w, poi, level=level,
            )
      obs_limit, exp_limit = norm*obs_limit, [ norm*lim for lim in exp_limit ]
      self.exp_limit = np.array(exp_limit)*300 # fb


class SixB():

   def __init__(self, filename, treename='sixBtree', year=2018, presel=False, gen=False, b_cutflow=False):
      print(filename)
      self.filename = re.search('NMSSM_.+/', filename).group()[:-1]
      tree = uproot.open(f"{filename}:{treename}")
      self.tree = tree
      self.nevents = len(tree['Run'].array())

      cutflow = uproot.open(f"{filename}:h_cutflow_unweighted")
      # save total number of events for scaling purposes
      total = cutflow.to_numpy()[0][0]
      self.scaled_total = total
      samp, xsec = next( ((key,value) for key,value in xsecMap.items() if key in filename),("unk",1) )
      self.sample = latexTitle(filename)
      self.xsec = xsec
      self.lumi = lumiMap[year][0]
      self.scale = self.lumi*xsec/total
      self.cutflow = (cutflow.to_numpy()[0]*self.scale).astype(int)
      self.cutflow_norm = (cutflow.to_numpy()[0]/cutflow.to_numpy()[0][0]).astype(int)
      self.mxmy = self.sample.replace('$','').replace('_','').replace('= ','_').replace(', ','_').replace(' GeV','')

      pattern = re.compile('H.+_') # Search for any keys beginning with an 'H' and followed somewhere by a '_'
      for k, v in tree.items():
         if re.match(pattern, k):
               setattr(self, k, v.array())


      def tree_keys(self):
         return self.tree.keys()

      def keys(self):
         return getattr(self, )


class Data():

   def __init__(self, filename, treename='sixBtree', year=2018, b_cutflow=False):
      tree = uproot.open(f"{filename}:{treename}")
      self.tree = tree
      self.cutflow = uproot.open(f"{filename}:h_cutflow_unweighted").to_numpy()[0]
      self.nevents = len(tree['Run'].array())

      for k, v in tree.items():
         if 'jet' in k or 'gen' in k:
            setattr(self, k, v.array())
      self.jet_higgsIdx = (self.jet_signalId + 2) // 2

      jets = ['HX_b1_', 'HX_b2_', 'H1_b1_', 'H1_b2_', 'H2_b1_', 'H2_b2_']
      self.btag_avg = np.column_stack(([self.get(jet + 'btag').to_numpy() for jet in jets])).sum(axis=1)/6

      self.initialize_vars()