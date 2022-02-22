"""
Store ROOT events in a standard and convenient way.

Classes:
    Signal
"""

from utils import *
from utils.varUtils import *
from utils.plotter import latexTitle
from .particle import Particle

# Standard library imports
# from math import comb
import re
import sys 
import uproot
import pandas as pd

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

    def __init__(self, filename, treename='sixBtree', year=2018):
        
        if 'NMSSM' in filename:
            self.filename = re.search('NMSSM_.+/', filename).group()[:-1]
        tree = uproot.open(f"{filename}:{treename}")
        self.tree = tree

        # Search for any keys beginning with an 'H' and followed somewhere by a '_'
        pattern = re.compile('H.+_')
        for k, v in tree.items():
            if re.match(pattern, k):
                setattr(self, k, v.array())

        jets = ['HX_b1_', 'HX_b2_', 'HY1_b1_', 'HY1_b2_', 'HY2_b1_', 'HY2_b2_']
        try: self.btag_avg = np.column_stack(([self.get(jet + 'DeepJet').to_numpy() for jet in jets])).sum(axis=1)/6
        except: self.btag_avg = np.column_stack(([self.get(jet + 'btag').to_numpy() for jet in jets])).sum(axis=1)/6

        self.nevents = len(tree['Run'].array())

        if 'NMSSM' in filename:
            cutflow = uproot.open(f"{filename}:h_cutflow")
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
            self._isdata = True

        self.initialize_vars()
        
    def initialize_vars(self):
        """Initialize variables that don't exist in the original ROOT tree."""
        HX_b1  = Particle(self, 'HX_b1')
        HX_b2  = Particle(self, 'HX_b2')
        HY1_b1 = Particle(self, 'HY1_b1')
        HY1_b2 = Particle(self, 'HY1_b2')
        HY2_b1 = Particle(self, 'HY2_b1')
        HY2_b2 = Particle(self, 'HY2_b2')

        HX = HX_b1 + HX_b2
        HY1 = HY1_b1 + HY1_b2
        HY2 = HY2_b1 + HY2_b2

        self.HX_dr = HX_b1.deltaR(HX_b2)
        self.HY1_dr = HY1_b1.deltaR(HY1_b2)
        self.HY2_dr = HY2_b1.deltaR(HY2_b2)

        self.HX_HY1_dEta = HX.deltaEta(HY1)
        self.HY1_HY2_dEta = HY1.deltaEta(HY2)
        self.HY2_HX_dEta = HY2.deltaEta(HX)

        X = HX + HY1 + HY2
        self.X_m = X.m

    def keys(self):
        return self.tree.keys()

    def get(self, key, library='ak'):
        """Returns the key.
        Use case: Sometimes keys are accessed directly, e.g. tree.key, but other times, the key may be accessed from a list of keys, e.g. tree.get(['key']). This function handles the latter case.
        """
        if key in self.tree.keys():
            return self.tree[key].array(library=library)
        else:
            return getattr(self, key)
    
    def np(self, key):
        """Returns the key as a numpy array."""
        return self.get(key, library='np')

    def rectangular_region(self, SR_edge=25, VR_edge=60, CR_edge=-1):
        higgs = ['HX_m', 'HY1_m', 'HY2_m']
        Dm_cand = np.column_stack(([abs(self.get(mH,'np') - 125) for mH in higgs]))
        SR_mask = ak.all(Dm_cand <= SR_edge, axis=1) # SR
        VR_mask = ak.all(Dm_cand > SR_edge, axis=1) & ak.all(Dm_cand <= VR_edge, axis=1) # VR
        CR_mask = ak.all(Dm_cand > VR_edge, axis=1) # CR
        
        score_cut = 0.66

        ls_mask = self.btag_avg < score_cut # ls
        hs_mask = self.btag_avg >= score_cut # hs

        self.CRls_mask = CR_mask & ls_mask
        self.CRhs_mask = CR_mask & hs_mask
        self.VRls_mask = VR_mask & ls_mask
        self.VRhs_mask = VR_mask & hs_mask
        self.SRls_mask = SR_mask & ls_mask
        self.SRhs_mask = SR_mask & hs_mask

        self._rect_region_bool = True

    def spherical_region(self, SR_edge=25, VR_edge=50):
        higgs = ['HX_m', 'HY1_m', 'HY2_m']
        Dm_cand = np.column_stack(([abs(self.get(mH,'np') - 125) for mH in higgs]))
        Dm_cand = Dm_cand * Dm_cand
        Dm_cand = Dm_cand.sum(axis=1)
        Dm_cand = np.sqrt(Dm_cand)
        A_SR_mask = Dm_cand <= SR_edge # Analysis SR
        A_CR_mask = (Dm_cand > SR_edge) & (Dm_cand <= VR_edge) # Analysis CR

        Dm_cand = np.column_stack(([abs(self.get(mH,'np') - 185) for mH in higgs]))
        Dm_cand = Dm_cand * Dm_cand
        Dm_cand = Dm_cand.sum(axis=1)
        Dm_cand = np.sqrt(Dm_cand)
        V_SR_mask = Dm_cand <= SR_edge # Validation SR
        V_CR_mask = (Dm_cand > SR_edge) & (Dm_cand <= VR_edge) # Validation CR

        score_cut = 0.66

        ls_mask = self.btag_avg < score_cut # ls
        hs_mask = self.btag_avg >= score_cut # hs

        self.A_CRls_mask = A_CR_mask & ls_mask
        self.A_CRhs_mask = A_CR_mask & hs_mask
        self.A_SRls_mask = A_SR_mask & ls_mask
        self.A_SRhs_mask = A_SR_mask & hs_mask

        self.V_CRls_mask = V_CR_mask & ls_mask
        self.V_CRhs_mask = V_CR_mask & hs_mask
        self.V_SRls_mask = V_SR_mask & ls_mask
        self.V_SRhs_mask = V_SR_mask & hs_mask

        self._sphere_region_bool = True

    def region_yield(self, definition='rect', normalized=False):

        if definition == 'rect':
            if not self._rect_region_bool: self.rectangular_region()

            CRls_yield = int(ak.sum(self.CRls_mask)*self.scale)
            CRhs_yield = int(ak.sum(self.CRhs_mask)*self.scale)
            VRls_yield = int(ak.sum(self.VRls_mask)*self.scale)
            VRhs_yield = int(ak.sum(self.VRhs_mask)*self.scale)
            SRls_yield = int(ak.sum(self.SRls_mask)*self.scale)
            SRhs_yield = int(ak.sum(self.SRhs_mask)*self.scale)

            if self._isdata:
                SRls_yield, SRhs_yield = 0, 0

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
            if not self._sphere_region_bool: self.spherical_region()

            A_CRls_yield = ak.sum(self.A_CRls_mask)
            A_CRhs_yield = ak.sum(self.A_CRhs_mask)
            A_SRls_yield = ak.sum(self.A_SRls_mask)
            A_SRhs_yield = ak.sum(self.A_SRhs_mask)

            V_CRls_yield = ak.sum(self.V_CRls_mask)
            V_CRhs_yield = ak.sum(self.V_CRhs_mask)
            V_SRls_yield = ak.sum(self.V_SRls_mask)
            V_SRhs_yield = ak.sum(self.V_SRhs_mask)

            if isdata:
                A_SRls_yield, A_SRhs_yield = 0, 0

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
            yields = [A_CRls_yield, A_CRhs_yield, A_SRls_yield, A_SRhs_yield, V_CRls_yield, V_CRhs_yield, V_SRls_yield, V_SRhs_yield]

            index = [(region, col) for region,col in zip(regions,cols)]

            index = pd.MultiIndex.from_tuples(index)

            yields = pd.Series(yields, index=index)
            print(yields)
