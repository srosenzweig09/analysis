"""
Store ROOT events in a standard and convenient way.

Classes:
    Signal
"""

from cmath import nan
from utils import *
from utils.varUtils import *
from utils.plotter import latexTitle
from .particle import Particle

# Standard library imports
from array import array
from colorama import Fore
from hep_ml import reweight
import re
import ROOT
import sys 
import uproot
import pandas as pd
from hep_ml.metrics_utils import ks_2samp_weighted

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

   def __init__(self, filename, treename='sixBtree', year=2018, presel=False):
      if 'NMSSM' in filename:
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
         cutflow = uproot.open(f"{filename}:h_cutflow")
         print(cutflow.axis().labels())
         # save total number of events for scaling purposes
         total = cutflow.to_numpy()[0][0]
         self.scaled_total = total
         samp, xsec = next( ((key,value) for key,value in xsecMap.items() if key in filename),("unk",1) )
         self.sample = latexTitle(filename)
         self.xsec = xsec
         self.lumi = lumiMap[year][0]
         self.scale = self.lumi*xsec/total
         self.cutflow = cutflow.to_numpy()[0]*self.scale
         self.mXmY = self.sample.replace('$','').replace('_','').replace('= ','_').replace(', ','_').replace(' GeV','')
         self._isdata = False
      else:
         self.scale = 1
         self._isdata = True

      if presel:
         for k, v in tree.items():
            if 'jet' in k or 'gen' in k:
               setattr(self, k, v.array())
      else:
         jets = ['HX_b1_', 'HX_b2_', 'HY1_b1_', 'HY1_b2_', 'HY2_b1_', 'HY2_b2_']
         try: self.btag_avg = np.column_stack(([self.get(jet + 'DeepJet').to_numpy() for jet in jets])).sum(axis=1)/6
         except: self.btag_avg = np.column_stack(([self.get(jet + 'btag').to_numpy() for jet in jets])).sum(axis=1)/6

         self.btag_avg2 = np.column_stack(([self.get(jet + 'btag').to_numpy()**2 for jet in jets])).sum(axis=1)/6

         self.initialize_vars()
        
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
      HX_b1  = Particle(self, 'HX_b1')
      HX_b2  = Particle(self, 'HX_b2')
      HY1_b1 = Particle(self, 'HY1_b1')
      HY1_b2 = Particle(self, 'HY1_b2')
      HY2_b1 = Particle(self, 'HY2_b1')
      HY2_b2 = Particle(self, 'HY2_b2')

      bs = [HX_b1, HX_b2, HY1_b1, HY1_b2, HY2_b1, HY2_b2]
      pair1 = [HX_b1]*5 + [HX_b2]*4 + [HY1_b1]*3 + [HY1_b2]*2 + [HY2_b1]
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

      self.pt6bsum = HX_b1.pt + HX_b2.pt + HY1_b1.pt + HY1_b2.pt + HY2_b1.pt + HY2_b2.pt

      HX = HX_b1 + HX_b2
      HY1 = HY1_b1 + HY1_b2
      HY2 = HY2_b1 + HY2_b2

      self.HX_dr = HX_b1.deltaR(HX_b2)
      self.HY1_dr = HY1_b1.deltaR(HY1_b2)
      self.HY2_dr = HY2_b1.deltaR(HY2_b2)

      self.HX_HY1_dEta = HX.deltaEta(HY1)
      self.HY1_HY2_dEta = HY1.deltaEta(HY2)
      self.HY2_HX_dEta = HY2.deltaEta(HX)

      self.HX_HY1_dPhi = HX.deltaPhi(HY1)
      self.HY1_HY2_dPhi = HY1.deltaPhi(HY2)
      self.HY2_HX_dPhi = HY2.deltaPhi(HX)

      self.HX_HY1_dR = HX.deltaR(HY1)
      self.HY1_HY2_dR = HY1.deltaR(HY2)
      self.HY2_HX_dR = HY2.deltaR(HX)

      self.HX_costheta = HX.cosTheta
      self.HY1_costheta = HY1.cosTheta
      self.HY2_costheta = HY2.cosTheta

      X = HX + HY1 + HY2
      self.X_m = X.m

   def rectangular_region(self, config):
      """Defines rectangular region masks."""
      SR_edge = float(config['rectangular']['maxSR'])
      VR_edge = float(config['rectangular']['maxVR'])
      CR_edge = float(config['rectangular']['maxCR'])
      if CR_edge == -1: CR_edge = 9999

      score_cut = float(config['score']['threshold'])

      self.SR_edge = SR_edge
      self.VR_edge = VR_edge
      self.CR_edge = CR_edge

      higgs = ['HX_m', 'HY1_m', 'HY2_m']
      Dm_cand = np.column_stack(([abs(self.get(mH,'np') - 125) for mH in higgs]))
      SR_mask = ak.all(Dm_cand <= SR_edge, axis=1) # SR
      VR_mask = ak.all(Dm_cand > SR_edge, axis=1) & ak.all(Dm_cand <= VR_edge, axis=1) # VR
      CR_mask = ak.all(Dm_cand > VR_edge, axis=1) & ak.all(Dm_cand <= CR_edge, axis=1) # CR

      ls_mask = self.btag_avg < score_cut # ls
      hs_mask = self.btag_avg >= score_cut # hs

      self.V_CRls_mask = CR_mask & ls_mask
      self.V_CRhs_mask = CR_mask & hs_mask
      self.V_SRls_mask = VR_mask & ls_mask
      self.V_SRhs_mask = VR_mask & hs_mask
      self.A_CRls_mask = np.zeros_like(CR_mask)
      self.A_CRhs_mask = np.zeros_like(CR_mask)
      self.A_SRls_mask = SR_mask & ls_mask
      self.A_SRhs_mask = SR_mask & hs_mask

      if self._isdata:
         print(f"   SR_edge   {int(SR_edge)}")
         print(f"   VR_edge   {int(VR_edge)}")
         print(f"   CR_edge   {int(CR_edge)}")
         print(" -----------------")
         print()
         
         # blinded_mask = VR_mask | CR_mask
         self.dat_mX_V_CRls = self.np('X_m')[self.V_CRls_mask]
         self.dat_mX_V_CRhs = self.np('X_m')[self.V_CRhs_mask]
         self.dat_mX_V_SRls = self.np('X_m')[self.V_SRls_mask]
         self.dat_mX_V_SRhs = self.np('X_m')[self.V_SRhs_mask]
         self.dat_mX_A_CRls = np.zeros_like(self.np('X_m'))
         self.dat_mX_A_CRhs = np.zeros_like(self.np('X_m'))
         self.dat_mX_A_SRls = self.np('X_m')[self.A_SRls_mask]
         # self.dat_mX_blinded = self.np('X_m')[blinded_mask]
      else:
         return self.np('X_m')[self.SRhs_mask]

      self._rect_region_bool = True

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

      higgs = ['HX_m', 'HY1_m', 'HY2_m']

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
      skip = ['HX_eta', 'HY1_eta', 'HY2_eta', 'HX_HY1_dEta', 'HY1_HY2_dEta', 'HY2_HX_dEta']
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