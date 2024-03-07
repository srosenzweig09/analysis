import numpy as np
import awkward as ak
from hep_ml import reweight

def make_bins(style):
    if style == 'linspace': return np.linspace
    elif style == 'arange': return np.arange  

def get_deltaM(higgs, center):
    deltaM = np.column_stack(([abs(m - center) for m in higgs]))
    deltaM = deltaM * deltaM
    deltaM = ak.sum(deltaM, axis=1)
    deltaM = np.sqrt(deltaM)
    return deltaM

def get_region_mask(higgs, center, sr_edge, cr_edge):
   deltaM = np.column_stack(([abs(m - val) for m,val in zip(higgs,center)]))
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

class BDTRegion():
    def __init__(self, tree):

        self.definitions = tree.config['definitions']
        self.shape = self.definitions['shape']
        self.ar_center = float(self.definitions['ARcenter'])
        self.sr_edge = float(self.definitions['SRedge'])
        self.cr_edge = float(self.definitions['CRedge'])
        self.vr_edge = float(self.definitions['VRedge'])

        self.avg_btag = tree.config['avg_btag']
        self.avg_btag_threshold = float(self.avg_btag['threshold'])

        self.bdtparams = tree.config['BDT']
        self.nestimators  = int(self.bdtparams['Nestimators'])
        self.learningrate = float(self.bdtparams['learningRate'])
        self.maxdepth     = int(self.bdtparams['maxDepth'])
        self.minleaves    = int(self.bdtparams['minLeaves'])
        self.gbsubsample  = float(self.bdtparams['GBsubsample'])
        self.randomstate  = int(self.bdtparams['randomState'])
        variables = self.bdtparams['variables']
        if isinstance(variables, str): variables = variables.split(", ")
        self.variables = variables

        self.plot_params = tree.config['plot']
        self.plot_style = self.plot_params['style']
        self.nedges = int(self.plot_params['edges'])
        self.steps = int(self.plot_params['steps'])
        self.minMX = int(self.plot_params['minMX'])
        self.maxMX = int(self.plot_params['maxMX'])

        self.mBins = make_bins(self.plot_style)(self.minMX, self.maxMX, self.nedges)
        self.x_mBins = (self.mBins[:-1] + self.mBins[1:])/2

        self.higgs = [tree.H1.m, tree.H2.m, tree.H3.m]
        try: higgs = [m.to_numpy() for m in higgs]
        except: pass

        self.ar_deltaM = get_deltaM(self.higgs, self.ar_center)
        self.asr_mask = self.ar_deltaM <= self.sr_edge
        if tree._is_signal: self.asr_avgbtag = tree.btag_avg[self.asr_mask]
        self.acr_mask = (self.ar_deltaM > self.sr_edge) & (self.ar_deltaM <= self.cr_edge) # Analysis CR
        if not tree._is_signal: self.acr_avgbtag = tree.btag_avg[self.acr_mask]
        self.ar_mask = self.asr_mask | self.acr_mask

        self.ls_mask = tree.btag_avg < self.avg_btag_threshold
        self.hs_mask = tree.btag_avg >= self.avg_btag_threshold

        self.acr_ls_mask = self.acr_mask & self.ls_mask
        self.acr_hs_mask = self.acr_mask & self.hs_mask
        self.asr_ls_mask = self.asr_mask & self.ls_mask
        self.asr_hs_mask = self.asr_mask & self.hs_mask
        if not tree._is_signal:
            self.blind_mask = ~self.asr_hs_mask
            self.asr_hs_mask = np.zeros_like(self.asr_mask)
        self.sr_mask = self.asr_hs_mask

        if self.shape == 'concentric': self.init_concentric()
        elif self.shape == 'diagonal': self.init_diagonal()
        elif self.shape == 'multiple': self.init_multiple()
        else: raise ValueError(f"Unknown shape: {self.shape}")

        self.sr_efficiency = ak.sum(self.asr_hs_mask)/len(self.asr_hs_mask)

        self.copy_attributes(tree)

    def init_concentric(self):
        self.vsr_mask = (self.ar_deltaM <= self.cr_edge) &  (self.ar_deltaM > self.sr_edge)
        self.vcr_mask = (self.ar_deltaM > self.cr_edge) & (self.ar_deltaM <= self.vr_edge)

        self.vcr_ls_mask = self.vcr_mask & self.ls_mask
        self.vcr_hs_mask = self.vcr_mask & self.hs_mask
        self.vsr_ls_mask = self.vsr_mask & self.ls_mask
        self.vsr_hs_mask = self.vsr_mask & self.hs_mask

        self.crtf = round(1 + np.sqrt(1/self.acr_ls_mask.to_numpy().sum()+1/self.acr_hs_mask.to_numpy().sum()),2)
        self.vrtf = 1 + np.sqrt(1/self.vcr_ls_mask.to_numpy().sum()+1/self.vcr_hs_mask.to_numpy().sum())

    def init_diagonal(self):
        self.vr_deltaM = get_deltaM(self.higgs, self.vr_center)
        self.vsr_mask = self.vr_deltaM <= self.sr_edge
        self.vcr_mask = (self.vr_deltaM > self.sr_edge) & (self.vr_deltaM <= self.cr_edge)

        self.vcr_ls_mask = self.vcr_mask & self.ls_mask
        self.vcr_hs_mask = self.vcr_mask & self.hs_mask
        self.vsr_ls_mask = self.vsr_mask & self.ls_mask
        self.vsr_hs_mask = self.vsr_mask & self.hs_mask

    def init_multiple(self):
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
        
        self.asr_mask, self.acr_mask = get_region_mask(self.higgs, (125,125,125), self.sr_edge, self.cr_edge)

        vsr_masks, vcr_masks = [], []
        self.vsr_mask = np.repeat(False, len(self.asr_mask))
        self.vcr_mask = np.repeat(False, len(self.asr_mask))
        for center in val_centers:
            vsr_mask, vcr_mask = get_region_mask(self.higgs, center, self.sr_edge, self.cr_edge)
            self.vsr_mask = np.logical_or(self.vsr_mask, vsr_mask)
            self.vcr_mask = np.logical_or(self.vcr_mask, vcr_mask)
            vsr_masks.append(vsr_mask)
            vcr_masks.append(vcr_mask)

        assert ak.any(self.vsr_mask), "No validation region found. :("

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

    def set_variables(self, var_list):
        self.variables = var_list

    def initialize_vars(self, tree):
        self.tight_mask = tree.jet_btag > tree.tight_wp
        medium_mask = tree.jet_btag > tree.medium_wp
        loose_mask = tree.jet_btag > tree.loose_wp

        self.fail_mask = ~loose_mask
        self.loose_mask = loose_mask & ~medium_mask
        self.medium_mask = medium_mask & ~self.tight_mask

        self.n_tight = ak.sum(self.tight_mask, axis=1)
        self.n_medium = ak.sum(self.medium_mask, axis=1)
        self.n_loose = ak.sum(self.loose_mask, axis=1)
        self.n_fail = ak.sum(self.fail_mask, axis=1)

        bs = [tree.H1.b1, tree.H1.b2, tree.H2.b1, tree.H2.b2, tree.H3.b1, tree.H3.b2]
        pair1 = [tree.H1.b1]*5 + [tree.H1.b2]*4 + [tree.H2.b1]*3 + [tree.H2.b2]*2 + [tree.H3.b1]
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

        self.pt6bsum = tree.H1.b1.pt + tree.H1.b2.pt + tree.H2.b1.pt + tree.H2.b2.pt + tree.H3.b1.pt + tree.H3.b2.pt

        self.H1_pt = tree.H1.pt
        self.H2_pt = tree.H2.pt
        self.H3_pt = tree.H3.pt

        self.H1_dr = tree.H1.dr
        self.H2_dr = tree.H2.dr
        self.H3_dr = tree.H3.dr

        self.H1_m = tree.H1.m
        self.H2_m = tree.H2.m
        self.H3_m = tree.H3.m

        self.H1_H2_dEta = abs(tree.H1.deltaEta(tree.H2))
        self.H2_H3_dEta = abs(tree.H2.deltaEta(tree.H3))
        self.H3_H1_dEta = abs(tree.H3.deltaEta(tree.H1))

        self.H1_H2_dPhi = tree.H1.deltaPhi(tree.H2)
        self.H2_H3_dPhi = tree.H2.deltaPhi(tree.H3)
        self.H3_H1_dPhi = tree.H3.deltaPhi(tree.H1)

        self.H1_H2_dr = tree.H1.deltaR(tree.H2)
        self.H1_H3_dr = tree.H3.deltaR(tree.H1)
        self.H2_H3_dr = tree.H2.deltaR(tree.H3)
        
        self.Y_H1_dr = tree.Y.deltaR(tree.H1)

        self.H1_costheta = abs(np.cos(tree.H1.P4.theta))
        self.H2_costheta = abs(np.cos(tree.H2.P4.theta))
        self.H3_costheta = abs(np.cos(tree.H3.P4.theta))

        self.H1_H2_dr = tree.H1.deltaR(tree.H2)
        self.H2_H3_dr = tree.H3.deltaR(tree.H2)
        self.H3_H1_dr = tree.H1.deltaR(tree.H3)

        self.X_m = tree.X.m

        self.X_m_prime = tree.X_m - tree.H1_m - tree.H2_m - tree.H3_m + 3*125

        self.H_j_btag = np.column_stack((
            tree.H1_b1_btag.to_numpy(),
            tree.H1_b2_btag.to_numpy(),
            tree.H2_b1_btag.to_numpy(),
            tree.H2_b2_btag.to_numpy(),
            tree.H3_b1_btag.to_numpy(),
            tree.H3_b2_btag.to_numpy()
        ))

        self.n_H_jet_tight = np.sum(self.H_j_btag >= tree.tight_wp, axis=1)
        self.n_H_jet_medium = np.sum(self.H_j_btag >= tree.medium_wp, axis=1)
        self.n_H_jet_loose = np.sum(self.H_j_btag >= tree.loose_wp, axis=1)

        self.copy_attributes(tree)

    def set_var_dict(self, tree):
        self.var_dict = {
            'pt6bsum' : tree.pt6bsum,
            'dR6bmin' : tree.dR6bmin,
            'dEta6bmax' : tree.dEta6bmax,
            'H1_m' : tree.H1.m,
            'H2_m' : tree.H2.m,
            'H3_m' : tree.H3.m,
            'H1_pt' : tree.H1.pt,
            'H2_pt' : tree.H2.pt,
            'H3_pt' : tree.H3.pt,
            'H1_dr' : tree.H1.dr,
            'H2_dr' : tree.H2.dr,
            'H3_dr' : tree.H3.dr,
            'H1_costheta' : tree.H1.costheta,
            'H2_costheta' : tree.H2.costheta,
            'H3_costheta' : tree.H3.costheta,
            'H1_H2_dr' : tree.H1.deltaR(tree.H2),
            'H2_H3_dr' : tree.H2.deltaR(tree.H3),
            'H3_H1_dr' : tree.H3.deltaR(tree.H1),
            'H1_H2_dEta' : tree.H1.deltaEta(tree.H2),
            'H2_H3_dEta' : tree.H2.deltaEta(tree.H3),
            'H3_H1_dEta' : tree.H3.deltaEta(tree.H1),
            'H1_H2_dPhi' : tree.H1.deltaPhi(tree.H2),
            'H2_H3_dPhi' : tree.H2.deltaPhi(tree.H3),
            'H3_H1_dPhi' : tree.H3.deltaPhi(tree.H1),
            'Y_m' : tree.Y.m,
        }

    def get_df(self, mask, variables):
      import pandas as pd
      features = {}
      for var in variables:
         # features[var] = abs(getattr(self, var)[mask])
         features[var] = abs(self.var_dict[var][mask])
      df = pd.DataFrame(features)
      return df

    def train(self, tree, sr_ls_mask, cr_hs_mask, cr_ls_mask, region):
        """
        region is either "ar" or "vr"
        """
        self.initialize_vars(tree)
        self.set_var_dict(tree)
        print(f"high avg b tag score threshold = {self.avg_btag_threshold}")
        
        tf = sum(self.acr_hs_mask)/sum(cr_ls_mask)
        ls_weights = np.ones(ak.sum(cr_ls_mask))*tf
        hs_weights = np.ones(ak.sum([cr_hs_mask]))

        df_ls = self.get_df(cr_ls_mask, self.variables)
        df_hs = self.get_df(cr_hs_mask, self.variables)

        np.random.seed(self.randomstate)
        reweighter_base = reweight.GBReweighter(
            n_estimators=self.nestimators, 
            learning_rate=self.learningrate, 
            max_depth=self.maxdepth, 
            min_samples_leaf=self.minleaves,
            gb_args={'subsample': self.gbsubsample})

        reweighter = reweight.FoldingReweighter(reweighter_base, random_state=self.randomstate, n_folds=2, verbose=False)

        reweighter.fit(df_ls,df_hs,ls_weights,hs_weights)

        df_ls = self.get_df(sr_ls_mask, self.variables)
        initial_weights = np.ones(ak.sum(sr_ls_mask))*tf

        sr_weights = reweighter.predict_weights(df_ls,initial_weights,lambda x: np.mean(x, axis=0))

        cr_df_ls = self.get_df(cr_ls_mask, self.variables)
        cr_initial_weights = np.ones(ak.sum(cr_ls_mask))*tf
        cr_weights = reweighter.predict_weights(cr_df_ls,cr_initial_weights,lambda x: np.mean(x, axis=0))

        if region == "ar":
            self.ar_tf = tf
            self.reweighter = reweighter
            self.asr_weights = sr_weights
            self.acr_weights = cr_weights
        elif region == "vr":
            self.vr_tf = tf
            self.vreweighter = reweighter
            self.vsr_weights = sr_weights
            self.vcr_weights = cr_weights

        self.copy_attributes(tree)

    def ks_test(self, tree, variable, ls_mask, hs_mask, weights):
      from scipy.stats import kstwobign
      from awkward.highlevel import Array

      ratio = 1

      var = getattr(tree, variable)

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

    def copy_attributes(self, dest_cls):
        # but not methods
        for attr_name, attr_value in vars(self).items():
            if not callable(attr_value): setattr(dest_cls, attr_name, attr_value)