"""
Project: 6b Final States - 
Author: Suzanne Rosenzweig

Classes:
Tree - Extracts information from ROOT TTrees.
TrainSix - Produces inputs to train the 6-jet classifier.
TrainTwo - A work in progress. Produces inputs to train the 2-jet classifier.

Notes:
Training samples are prepared such that these requirements are already imposed:
- n_jet > 6
- n_sixb == 6
"""

from . import *
from .modelUtils.load import load_model
from .varUtils import *
from .plotter import latexTitle

# Standard library imports
# from math import comb
import re
import sys 
import uproot

vector.register_awkward()

def get_scaled_weights(list_of_arrs, bins, scale):
    """This function is used to get weights for background events."""
    n = np.zeros_like(bins[:-1])
    for i,(sample, scale) in enumerate(zip(list_of_arrs, scale)):
        try: branch = sample.to_numpy()
        except: branch = ak.flatten(sample).to_numpy()
        n_i, b = np.histogram(branch, bins)
        centers = (b[1:] + b[:-1])/2
        # print(n_i)
        # print(scale)
        try: n += n_i*scale
        except: n += n_i
    return n, b, centers

def build_p4(pt, eta, phi, m):
    return vector.obj(pt=pt, eta=eta, phi=phi, m=m)

def p4(tree, jet):
    return  vector.obj(pt=tree.jet_pt[jet],
                        eta=tree.jet_eta[jet],
                        phi=tree.jet_phi[jet],
                        m=tree.jet_m[jet])

def get_boosted_p4(tree, jets):
    jet_p4 = p4(tree, jets[0])
    for jet in jets[1:]:
        jet_p4 += p4(tree, jet)
    
    jet_pt = []
    for jet in jets:
        jet_pt.append(ak.flatten(p4(tree, jet).boost_p4(jet_p4).pt).to_numpy()[:, np.newaxis])

    return jet_pt

def get_6jet_p4(p4):
    combos = ak.combinations(p4, 6)
    part0, part1, part2, part3, part4, part5 = ak.unzip(combos)
    evt_p4 = part0 + part1 + part2 + part3 + part4 + part5
    boost_0 = part0.boost_p4(evt_p4)
    boost_1 = part1.boost_p4(evt_p4)
    boost_2 = part2.boost_p4(evt_p4)
    boost_3 = part3.boost_p4(evt_p4)
    boost_4 = part4.boost_p4(evt_p4)
    boost_5 = part5.boost_p4(evt_p4)
    return evt_p4, [boost_0, boost_1, boost_2, boost_3, boost_4, boost_5]

# class Data():


class Signal():
    def __init__(self, filename, treename='sixBtree', year=2018, training=False):
        """
        A class for handling TTrees.

        args:
            filename: string containing name of a single file OR a list of several files to open
            treename: default is 'sixBtree,' which is the named TTree output from the analysis code

        returns:
            Nothing. Initializes attributes of Tree class.
        """
        
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
        # elif 'Data' in filename:

        higgs = ['HX_m', 'HY1_m', 'HY2_m']
        self.DeltaM = np.column_stack(([abs(self.get(mH,'np') - 125) for mH in higgs]))
        self.mCand = np.column_stack(([abs(self.get(mH,'np')) for mH in higgs]))
        
        self.DeltaM_V = np.column_stack(([abs(self.get(mH,'np') - 185) for mH in higgs]))

    def keys(self):
        print(self.tree.keys())

    def get(self, key, library='ak'):
        return self.tree[key].array(library=library)

    def region_yield(self, normalized=False):

        # edges of mH windows
        SR_edge = 25 # GeV
        VR_edge = 60 # GeV
        CR_edge = 120 # GeV

        mH = 125 # GeV

        higgs = ['HX_m', 'HY1_m', 'HY2_m']
        Dm_cand = np.column_stack(([abs(self.get(mH,'np') - 125) for mH in higgs]))
        SR = ak.all(Dm_cand <= SR_edge, axis=1) # SR
        VR = ak.all(Dm_cand > SR_edge, axis=1) & ak.all(Dm_cand <= VR_edge, axis=1) # VR
        CR = ak.all(Dm_cand > VR_edge, axis=1) # CR

        score_cut = 0.66

        ls_mask = self.btag_avg < score_cut # ls
        hs_mask = self.btag_avg >= score_cut # hs

        if normalized:
            CRls_yield = ak.sum(CR & ls_mask)*self.scale
            CRhs_yield = ak.sum(CR & hs_mask)*self.scale
            VRls_yield = ak.sum(VR & ls_mask)*self.scale
            VRhs_yield = ak.sum(VR & hs_mask)*self.scale
            SRls_yield = ak.sum(SR & ls_mask)*self.scale
            SRhs_yield = ak.sum(SR & hs_mask)*self.scale

            total = CRls_yield + CRhs_yield + VRls_yield + VRhs_yield + SRls_yield + SRhs_yield
            
            CRls_yield = int((CRls_yield / total)*100)
            CRhs_yield = int((CRhs_yield / total)*100)
            VRls_yield = int((VRls_yield / total)*100)
            VRhs_yield = int((VRhs_yield / total)*100)
            SRls_yield = int((SRls_yield / total)*100)
            SRhs_yield = int((SRhs_yield / total)*100)

        else:
            CRls_yield = int(ak.sum(CR & ls_mask)*self.scale)
            CRhs_yield = int(ak.sum(CR & hs_mask)*self.scale)
            VRls_yield = int(ak.sum(VR & ls_mask)*self.scale)
            VRhs_yield = int(ak.sum(VR & hs_mask)*self.scale)
            SRls_yield = int(ak.sum(SR & ls_mask)*self.scale)
            SRhs_yield = int(ak.sum(SR & hs_mask)*self.scale)

        print('|Signal Region | Validation Region | Control Region|')
        print('|Low   | High  | Low     |    High | Low    |  High|')
        print(f'|{SRls_yield}  | {SRhs_yield}  | {VRls_yield}     | {VRhs_yield}     | {CRls_yield}    | {CRhs_yield}  |')

    def region_yield_sphere(self, normalized=False):

        # edges of mH windows
        SR_edge = 25 # GeV
        VR_edge = 60 # GeV
        CR_edge = 120 # GeV

        mH = 125 # GeV

        higgs = ['HX_m', 'HY1_m', 'HY2_m']
        Dm_cand = np.column_stack(([abs(self.get(mH,'np') - 125) for mH in higgs]))
        Dm_cand = Dm_cand * Dm_cand
        Dm_cand = Dm_cand.sum(axis=1)
        Dm_cand = np.sqrt(Dm_cand)
        A_SR = Dm_cand <= SR_edge # SR
        A_CR = (Dm_cand > SR_edge) & (Dm_cand <= VR_edge) # VR

        Dm_cand = np.column_stack(([abs(self.get(mH,'np') - 185) for mH in higgs]))
        Dm_cand = Dm_cand * Dm_cand
        Dm_cand = Dm_cand.sum(axis=1)
        Dm_cand = np.sqrt(Dm_cand)
        V_SR = Dm_cand <= SR_edge # SR
        V_CR = (Dm_cand > SR_edge) & (Dm_cand <= VR_edge) # VR

        score_cut = 0.66

        ls_mask = self.btag_avg < score_cut # ls
        hs_mask = self.btag_avg >= score_cut # hs

        if normalized:
            A_CRls_yield = ak.sum(A_CR & ls_mask)
            A_CRhs_yield = ak.sum(A_CR & hs_mask)
            A_SRls_yield = ak.sum(A_SR & ls_mask)
            A_SRhs_yield = ak.sum(A_SR & hs_mask)

            A_total = A_CRls_yield + A_CRhs_yield + A_SRls_yield + A_SRhs_yield

            V_CRls_yield = ak.sum(V_CR & ls_mask)
            V_CRhs_yield = ak.sum(V_CR & hs_mask)
            V_SRls_yield = ak.sum(V_SR & ls_mask)
            V_SRhs_yield = ak.sum(V_SR & hs_mask)

            V_total = V_CRls_yield + V_CRhs_yield + V_SRls_yield + V_SRhs_yield

            A_CRls_yield = A_CRls_yield / A_total
            A_CRhs_yield = A_CRhs_yield / A_total
            A_SRls_yield = A_SRls_yield / A_total
            A_SRhs_yield = A_SRhs_yield / A_total
            
            V_CRls_yield = V_CRls_yield / V_total
            V_CRhs_yield = V_CRhs_yield / V_total
            V_SRls_yield = V_SRls_yield / V_total
            V_SRhs_yield = V_SRhs_yield / V_total

        else:
            A_CRls_yield = int(ak.sum(A_CR & ls_mask)*self.scale)
            A_CRhs_yield = int(ak.sum(A_CR & hs_mask)*self.scale)
            A_SRls_yield = int(ak.sum(A_SR & ls_mask)*self.scale)
            A_SRhs_yield = int(ak.sum(A_SR & hs_mask)*self.scale)

            V_CRls_yield = int(ak.sum(V_CR & ls_mask)*self.scale)
            V_CRhs_yield = int(ak.sum(V_CR & hs_mask)*self.scale)
            V_SRls_yield = int(ak.sum(V_SR & ls_mask)*self.scale)
            V_SRhs_yield = int(ak.sum(V_SR & hs_mask)*self.scale)

        print('|Signal Region | Validation Region | Control Region|')
        print('|Low   | High  | Low     |    High | Low    |  High|')
        print(f'|{SRls_yield}  | {SRhs_yield}  | {VRls_yield}     | {VRhs_yield}     | {CRls_yield}    | {CRhs_yield}  |')


class Tree():
    def __init__(self, filename, treename='sixBtree', year=2018, training=False, exploration=False, signal=True):
        """
        A class for handling TTrees.

        args:
            filename: string containing name of a single file OR a list of several files to open
            treename: default is 'sixBtree,' which is the named TTree output from the analysis code

        returns:
            Nothing. Initializes attributes of Tree class.
        """

        if type(filename) != list:
            self.single_init(filename, treename, year, training=training, exploration=exploration, signal=signal)
        else:
            self.multi_init(filename, treename, year)

    def single_init(self, filename, treename, year, training, exploration, signal):
        """Opens a single file into a TTree"""
        tree = uproot.open(f"{filename}:{treename}")
        self.tree = tree

        if not exploration:
            for k, v in tree.items():
                if ('t6' in k) or ('score' in k) or ('nn' in k): 
                    setattr(self, k, v.array())
                    used_key = k
        else:            
            for k, v in tree.items():
                if (k.startswith('jet_') or (k.startswith('gen_'))):
                    setattr(self, k, v.array())
                    used_key = k

        self.nevents = len(tree[used_key].array())

        # for k, v in tree.items():
        #     setattr(self, k, v.array())
        # nevents = len(tree[k].array())

        # self.local_ind = ak.local_index(self.jet_pt)

        # try:
        #     self.signal_evt_mask = ak.sum(self.jet_signalId > -1, axis=1) == 6
        #     self.signal_jet_mask = self.jet_signalId[self.signal_evt_mask] > -1
        # except:
        #     self.signal_evt_mask = ak.sum(self.jet_idx > -1, axis=1) == 6
        #     self.signal_jet_mask = self.jet_idx[self.signal_evt_mask] > -1

        if training: return
        cutflow = uproot.open(f"{filename}:h_cutflow")
        # save total number of events for scaling purposes
        total = cutflow.to_numpy()[0][0]
        _, xsec = next( ((key,value) for key,value in xsecMap.items() if key in filename),("unk",1) )
        if signal: 
            self.sample = latexTitle(filename)
            self.mXmY = self.sample.replace('$','').replace('_','').replace('= ','_').replace(', ','_').replace(' GeV','')
        self.xsec = xsec
        self.lumi = lumiMap[year][0]
        self.scale = self.lumi*xsec/total
        self.cutflow = cutflow.to_numpy()[0]*self.scale

        # for k, v in tree.items():
        #     try: setattr(self, 'signal_' + k, v.array()[self.signal_evt_mask][self.signal_jet_mask])
        #     except: continue

    # def get_sr_cr(self):
        # """Triple mass veto."""

    def multi_init(self, filelist, treename, year=2018):
        """Opens a list of files into several TTrees. Typically used for bkgd events."""
        self.is_signal = False
        self.is_bkgd = True
        trees = []
        xsecs = []
        nevent = []
        sample = []
        total = []
        self.cutflow = np.zeros(11)
        for filename in filelist:
            # Open tree
            tree = uproot.open(f"{filename}:{treename}")
            # How many events in the tree?
            n = len(tree['n_jet'].array())
            cutflow = uproot.open(f"{filename}:h_cutflow")
            # save total number of events for scaling purposes
            samp_tot = cutflow.to_numpy()[0][0]
            cf = cutflow.to_numpy()[0]
            if len(cf) < 11: cf = np.append(cf, np.zeros(11-len(cf)))
            if n == 0: continue # Skip if no events
            total.append(samp_tot)
            nevent.append(n)
            # Bkgd events must be scaled to the appropriate xs in order to compare fairly with signal yield
            samp, xsec = next( ((key,value) for key,value in xsecMap.items() if key in filename),("unk",1) )
            self.cutflow += cf[:11] * lumiMap[year][0] * xsec / samp_tot
            trees.append(tree)
            xsecs.append(xsec)
            sample.append(samp)
            setattr(self, samp, tree)
            for k, v in tree.items():
                setattr(getattr(self, samp), k, v.array())
            
        self.ntrees = len(trees)
        self.tree = trees
        self.xsec = xsecs
        self.lumi = lumiMap[year][0]
        self.nevents = nevent
        self.sample = sample
        self.total = total
        self.scale = self.lumi*np.asarray(xsecs)/np.asarray(total)
        self.weighted_n = np.asarray(self.nevents)*self.scale

        for k in tree.keys():
            leaves = []
            for samp in self.sample:
                arr = getattr(getattr(self, samp), k)
                leaves.append(arr)
            setattr(self, k, leaves)

    def get_hist_weights(self, key, bins):
        if self.is_bkgd: 
            n = np.zeros_like(bins[:-1])
            for i,(sample, scale) in enumerate(zip(self.sample, self.scale)):
                try:
                    branch = ak.flatten(getattr(getattr(self, sample), key)).to_numpy()
                except:
                    branch = getattr(getattr(self, sample), key).to_numpy()
                n_i, b = np.histogram(branch, bins)
                centers = (b[1:] + b[:-1])/2
                n += n_i*scale
            return n, b, centers
        else: 
            branch = ak.flatten(getattr(self, key)).to_numpy()
            n, b = np.histogram(branch, bins)
            centers = (b[1:] + b[:-1])/2
            # n_tot = sum(n)
            # epsilon = 5e-3
            # max_bin = sum(n > epsilon*n_tot) + 1
            # new_bins = np.linspace(branch.min(), bins[max_bin], 100)
            # n, b = np.histogram(branch, new_bins)
            # centers = (b[1:] + b[:-1])/2
        return n*self.scale, b, centers

    def initialize_reco_p4(self):
        # particles = ['gen_HX_b1','gen_HX_b2','gen_HY1_b1','gen_HY1_b2','gen_HY2_b1','gen_HY2_b2']

        self.HX_b1_p4 = vector.obj(
            pt  = self.get('gen_HX_b1_recojet_pt'),
            eta = self.get('gen_HX_b1_recojet_eta'),
            phi = self.get('gen_HX_b1_recojet_phi'),
            m   = self.get('gen_HX_b1_recojet_m'))

        self.HX_b2_p4 = vector.obj(
            pt  = self.get('gen_HX_b2_recojet_pt'),
            eta = self.get('gen_HX_b2_recojet_eta'),
            phi = self.get('gen_HX_b2_recojet_phi'),
            m   = self.get('gen_HX_b2_recojet_m'))

        self.HY1_b1_p4 = vector.obj(
            pt  = self.get('gen_HY1_b1_recojet_pt'),
            eta = self.get('gen_HY1_b1_recojet_eta'),
            phi = self.get('gen_HY1_b1_recojet_phi'),
            m   = self.get('gen_HY1_b1_recojet_m'))

        self.HY1_b2_p4 = vector.obj(
            pt  = self.get('gen_HY1_b2_recojet_pt'),
            eta = self.get('gen_HY1_b2_recojet_eta'),
            phi = self.get('gen_HY1_b2_recojet_phi'),
            m   = self.get('gen_HY1_b2_recojet_m'))

        self.HY2_b1_p4 = vector.obj(
            pt  = self.get('gen_HY2_b1_recojet_pt'),
            eta = self.get('gen_HY2_b1_recojet_eta'),
            phi = self.get('gen_HY2_b1_recojet_phi'),
            m   = self.get('gen_HY2_b1_recojet_m'))

        self.HY2_b2_p4 = vector.obj(
            pt  = self.get('gen_HY2_b2_recojet_pt'),
            eta = self.get('gen_HY2_b2_recojet_eta'),
            phi = self.get('gen_HY2_b2_recojet_phi'),
            m   = self.get('gen_HY2_b2_recojet_m'))

    def initialize_t6_X(self):
        self.t6_H1 = vector.obj(
            pt  = self.t6_higgs_pt[:,0],
            eta = self.t6_higgs_eta[:,0],
            phi = self.t6_higgs_phi[:,0],
            m   = self.t6_higgs_m[:,0])

        self.t6_H2 = vector.obj(
            pt  = self.t6_higgs_pt[:,1],
            eta = self.t6_higgs_eta[:,1],
            phi = self.t6_higgs_phi[:,1],
            m   = self.t6_higgs_m[:,1])

        self.t6_H3 = vector.obj(
            pt  = self.t6_higgs_pt[:,2],
            eta = self.t6_higgs_eta[:,2],
            phi = self.t6_higgs_phi[:,2],
            m   = self.t6_higgs_m[:,2])

        self.t6_X = self.t6_H1 + self.t6_H2 + self.t6_H3


    def initialize_t6_X_multi(self):
        self.t6_H1 = vector.obj(
            pt  = self.t6_higgs_pt[:,0],
            eta = self.t6_higgs_eta[:,0],
            phi = self.t6_higgs_phi[:,0],
            m   = self.t6_higgs_m[:,0])

        self.t6_H2 = vector.obj(
            pt  = self.t6_higgs_pt[:,1],
            eta = self.t6_higgs_eta[:,1],
            phi = self.t6_higgs_phi[:,1],
            m   = self.t6_higgs_m[:,1])

        self.t6_H3 = vector.obj(
            pt  = self.t6_higgs_pt[:,2],
            eta = self.t6_higgs_eta[:,2],
            phi = self.t6_higgs_phi[:,2],
            m   = self.t6_higgs_m[:,2])

        self.t6_X = self.t6_H1 + self.t6_H2 + self.t6_H3

    def initialize_t6_p4(self):
        # particles = ['gen_HX_b1','gen_HX_b2','gen_HY1_b1','gen_HY1_b2','gen_HY2_b1','gen_HY2_b2']

        higgs1_mask = self.t6_jet_higgsIdx == 0
        Hb1_mask = ak.argsort(self.t6_jet_pt[higgs1_mask], axis=1)
        higgs2_mask = self.t6_jet_higgsIdx == 1
        Hb2_mask = ak.argsort(self.t6_jet_pt[higgs2_mask], axis=1)
        higgs3_mask = self.t6_jet_higgsIdx == 2
        Hb3_mask = ak.argsort(self.t6_jet_pt[higgs3_mask], axis=1)

        self.t6_H1_b1 = vector.obj(
            pt  = self.t6_jet_pt[higgs1_mask][Hb1_mask[:,0]],
            eta = self.t6_jet_eta[higgs1_mask][Hb1_mask[:,0]],
            phi = self.t6_jet_phi[higgs1_mask][Hb1_mask[:,0]],
            m   = self.t6_jet_m[higgs1_mask][Hb1_mask[:,0]])

        self.t6_H1_b2 = vector.obj(
            pt  = self.t6_jet_pt[higgs1_mask][Hb1_mask[:,1]],
            eta = self.t6_jet_eta[higgs1_mask][Hb1_mask[:,1]],
            phi = self.t6_jet_phi[higgs1_mask][Hb1_mask[:,1]],
            m   = self.t6_jet_m[higgs1_mask][Hb1_mask[:,1]])

        self.t6_H2_b1 = vector.obj(
            pt  = self.t6_jet_pt[higgs2_mask][Hb2_mask[:,0]],
            eta = self.t6_jet_eta[higgs2_mask][Hb2_mask[:,0]],
            phi = self.t6_jet_phi[higgs2_mask][Hb2_mask[:,0]],
            m   = self.t6_jet_m[higgs2_mask][Hb2_mask[:,0]])

        self.t6_H2_b2 = vector.obj(
            pt  = self.t6_jet_pt[higgs2_mask][Hb2_mask[:,1]],
            eta = self.t6_jet_eta[higgs2_mask][Hb2_mask[:,1]],
            phi = self.t6_jet_phi[higgs2_mask][Hb2_mask[:,1]],
            m   = self.t6_jet_m[higgs2_mask][Hb2_mask[:,1]])

        self.t6_H3_b1 = vector.obj(
            pt  = self.t6_jet_pt[higgs3_mask][Hb3_mask[:,0]],
            eta = self.t6_jet_eta[higgs3_mask][Hb3_mask[:,0]],
            phi = self.t6_jet_phi[higgs3_mask][Hb3_mask[:,0]],
            m   = self.t6_jet_m[higgs3_mask][Hb3_mask[:,0]])

        self.t6_H3_b2 = vector.obj(
            pt  = self.t6_jet_pt[higgs3_mask][Hb3_mask[:,1]],
            eta = self.t6_jet_eta[higgs3_mask][Hb3_mask[:,1]],
            phi = self.t6_jet_phi[higgs3_mask][Hb3_mask[:,1]],
            m   = self.t6_jet_m[higgs3_mask][Hb3_mask[:,1]])

        self.t6_X = self.t6_H1_b1 + self.t6_H1_b2 + self.t6_H2_b1 + self.t6_H2_b2 + self.t6_H3_b1 + self.t6_H3_b2

    # def get_t6_pair_p4(self):
    #     # if signal: mask = self.jet_idx > -1
    #     # else: mask = self.jet_idx == -1
    #     jet1_pt, jet2_pt = ak.unzip(ak.combinations(self.jet_pt[:,:6], 2))
    #     jet1_eta, jet2_eta = ak.unzip(ak.combinations(self.jet_eta[:,:6], 2))
    #     jet1_phi, jet2_phi = ak.unzip(ak.combinations(self.jet_phi[:,:6], 2))
    #     jet1_m, jet2_m = ak.unzip(ak.combinations(self.jet_m[:,:6], 2))
    #     jet1_idx, jet2_idx = ak.unzip(ak.combinations(self.jet_signalId[:,:6], 2))
    #     jet12_ind = ak.combinations(ak.local_index(self.jet_pt[:,:6]),2)
    #     jet1_ind, jet2_ind = ak.unzip(jet12_ind)

    #     jet1_p4 = vector.obj(
    #         pt  = jet1_pt,
    #         eta = jet1_eta,
    #         phi = jet1_phi,
    #         m   = jet1_m)

    #     jet2_p4 = vector.obj(
    #         pt  = jet2_pt,
    #         eta = jet2_eta,
    #         phi = jet2_phi,
    #         m   = jet2_m)

    #     return jet1_p4, jet2_p4, jet1_ind, jet2_ind, jet12_ind, jet1_idx, jet2_idx
    
    # def p4(self, pt, eta, phi, m):
    #     return vector.obj(pt=pt, eta=eta, phi=phi, m=m)

    # def get_all_p4(self, pts, etas, phis, ms):
    #     return [self.p4(pt, eta, phi, m) for pt,eta,phi,m in zip(pts,etas,phis,ms)]

    # def get_six_p4(self, mask=None):
    #     jet_pt = ak.unzip(ak.combinations(self.jet_pt[mask], 6))
    #     jet_eta = ak.unzip(ak.combinations(self.jet_eta[mask], 6))
    #     jet_phi = ak.unzip(ak.combinations(self.jet_phi[mask], 6))
    #     jet_m = ak.unzip(ak.combinations(self.jet_m[mask], 6))

    #     return self.get_all_p4(jet_pt, jet_eta, jet_phi, jet_m)

    # def sort(self, mask):
    #     for k, v in self.tree.items():
    #         try: setattr(self, k, v.array()[mask])
    #         except: continue

    def keys(self):
        print(self.tree.keys())

    def get(self, key):
        return self.tree[key].array()

# class TrainSix():
    
#     def get_boosted(self, tree, ind_array):

#         jet0_p4 = build_p4(tree.jet_pt[ind_array][:,0], 
#                            tree.jet_eta[ind_array][:,0], 
#                            tree.jet_phi[ind_array][:,0], 
#                            tree.jet_m[ind_array][:,0])
#         jet1_p4 = build_p4(tree.jet_pt[ind_array][:,1], 
#                            tree.jet_eta[ind_array][:,1], 
#                            tree.jet_phi[ind_array][:,1], 
#                            tree.jet_m[ind_array][:,1])
#         jet2_p4 = build_p4(tree.jet_pt[ind_array][:,2], 
#                            tree.jet_eta[ind_array][:,2], 
#                            tree.jet_phi[ind_array][:,2], 
#                            tree.jet_m[ind_array][:,2])
#         jet3_p4 = build_p4(tree.jet_pt[ind_array][:,3], 
#                            tree.jet_eta[ind_array][:,3], 
#                            tree.jet_phi[ind_array][:,3], 
#                            tree.jet_m[ind_array][:,3])
#         jet4_p4 = build_p4(tree.jet_pt[ind_array][:,4], 
#                            tree.jet_eta[ind_array][:,4], 
#                            tree.jet_phi[ind_array][:,4], 
#                            tree.jet_m[ind_array][:,4])
#         jet5_p4 = build_p4(tree.jet_pt[ind_array][:,5], 
#                            tree.jet_eta[ind_array][:,5], 
#                            tree.jet_phi[ind_array][:,5], 
#                            tree.jet_m[ind_array][:,5])

#         jet6_p4 = jet0_p4 + jet1_p4 + jet2_p4 + jet3_p4 + jet4_p4 + jet5_p4

#         jet0_boost = jet0_p4.boost_p4(jet6_p4)
#         jet1_boost = jet1_p4.boost_p4(jet6_p4)
#         jet2_boost = jet2_p4.boost_p4(jet6_p4)
#         jet3_boost = jet3_p4.boost_p4(jet6_p4)
#         jet4_boost = jet4_p4.boost_p4(jet6_p4)
#         jet5_boost = jet5_p4.boost_p4(jet6_p4)

#         # jet0_boosted_pt = jet0_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
#         # jet1_boosted_pt = jet1_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
#         # jet2_boosted_pt = jet2_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
#         # jet3_boosted_pt = jet3_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
#         # jet4_boosted_pt = jet4_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
#         # jet5_boosted_pt = jet5_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]

#         return jet0_boost, jet1_boost, jet2_boost, jet3_boost, jet4_boost, jet5_boost, jet6_p4

#     def get_p4s(self, tree, ind_array):

#         jet0_p4 = build_p4(tree.jet_pt[ind_array][:,0], 
#                            tree.jet_eta[ind_array][:,0], 
#                            tree.jet_phi[ind_array][:,0], 
#                            tree.jet_m[ind_array][:,0])
#         jet1_p4 = build_p4(tree.jet_pt[ind_array][:,1], 
#                            tree.jet_eta[ind_array][:,1], 
#                            tree.jet_phi[ind_array][:,1], 
#                            tree.jet_m[ind_array][:,1])
#         jet2_p4 = build_p4(tree.jet_pt[ind_array][:,2], 
#                            tree.jet_eta[ind_array][:,2], 
#                            tree.jet_phi[ind_array][:,2], 
#                            tree.jet_m[ind_array][:,2])
#         jet3_p4 = build_p4(tree.jet_pt[ind_array][:,3], 
#                            tree.jet_eta[ind_array][:,3], 
#                            tree.jet_phi[ind_array][:,3], 
#                            tree.jet_m[ind_array][:,3])
#         jet4_p4 = build_p4(tree.jet_pt[ind_array][:,4], 
#                            tree.jet_eta[ind_array][:,4], 
#                            tree.jet_phi[ind_array][:,4], 
#                            tree.jet_m[ind_array][:,4])
#         jet5_p4 = build_p4(tree.jet_pt[ind_array][:,5], 
#                            tree.jet_eta[ind_array][:,5], 
#                            tree.jet_phi[ind_array][:,5], 
#                            tree.jet_m[ind_array][:,5])

#         jet6_p4 = jet0_p4 + jet1_p4 + jet2_p4 + jet3_p4 + jet4_p4 + jet5_p4

#         return jet0_p4, jet1_p4, jet2_p4, jet3_p4, jet4_p4, jet5_p4

#     def __init__(self, filename, dijet=False):

#         tree = Tree(filename, 'sixBtree', training=True)
#         self.tree = tree
#         nevents = len(tree.jet_pt)

#         n_sixb = tree.n_sixb
#         local_ind = ak.local_index(tree.jet_idx)
#         signal_jet_mask = tree.jet_idx > -1
#         signal_jet_ind  = local_ind[signal_jet_mask]
#         excess_jet_ind  = local_ind[~signal_jet_mask]
#         mixed_ind = ak.sort(ak.concatenate((excess_jet_ind, signal_jet_ind), axis=1)[:, :6], axis=1)
#         mixed_ind_np = mixed_ind.to_numpy()
#         excess_signal_mask = tree.jet_idx[mixed_ind] > -1

#         signal_p4 = vector.obj(
#             pt  = tree.jet_pt[signal_jet_ind],
#             eta = tree.jet_eta[signal_jet_ind],
#             phi = tree.jet_phi[signal_jet_ind],
#             m   = tree.jet_m[signal_jet_ind]
#         )

#         excess_p4 = vector.obj(
#             pt  = tree.jet_pt[mixed_ind],
#             eta = tree.jet_eta[mixed_ind],
#             phi = tree.jet_phi[mixed_ind],
#             m   = tree.jet_m[mixed_ind]
#         )

#         signal_btag = tree.jet_btag[signal_jet_ind].to_numpy()
#         excess_btag = tree.jet_btag[mixed_ind].to_numpy()


#         s_jet0_boost, s_jet1_boost, s_jet2_boost, s_jet3_boost, s_jet4_boost, s_jet5_boost, s_jet6_p4 = self.get_boosted(tree, signal_jet_mask)
#         e_jet0_boost, e_jet1_boost, e_jet2_boost, e_jet3_boost, e_jet4_boost, e_jet5_boost, e_jet6_p4 = self.get_boosted(tree, mixed_ind)

#         xtra_signal_features = np.column_stack((
#             s_jet0_boost.pt.to_numpy()[:,np.newaxis], 
#             s_jet1_boost.pt.to_numpy()[:,np.newaxis], 
#             s_jet2_boost.pt.to_numpy()[:,np.newaxis], 
#             s_jet3_boost.pt.to_numpy()[:,np.newaxis], 
#             s_jet4_boost.pt.to_numpy()[:,np.newaxis], 
#             s_jet5_boost.pt.to_numpy()[:,np.newaxis],
#             s_jet0_boost.eta.to_numpy()[:,np.newaxis], 
#             s_jet1_boost.eta.to_numpy()[:,np.newaxis], 
#             s_jet2_boost.eta.to_numpy()[:,np.newaxis], 
#             s_jet3_boost.eta.to_numpy()[:,np.newaxis], 
#             s_jet4_boost.eta.to_numpy()[:,np.newaxis], 
#             s_jet5_boost.eta.to_numpy()[:,np.newaxis],
#             s_jet0_boost.phi.to_numpy()[:,np.newaxis], 
#             s_jet1_boost.phi.to_numpy()[:,np.newaxis], 
#             s_jet2_boost.phi.to_numpy()[:,np.newaxis], 
#             s_jet3_boost.phi.to_numpy()[:,np.newaxis], 
#             s_jet4_boost.phi.to_numpy()[:,np.newaxis], 
#             s_jet5_boost.m.to_numpy()[:,np.newaxis],
#             s_jet0_boost.m.to_numpy()[:,np.newaxis], 
#             s_jet1_boost.m.to_numpy()[:,np.newaxis], 
#             s_jet2_boost.m.to_numpy()[:,np.newaxis], 
#             s_jet3_boost.m.to_numpy()[:,np.newaxis], 
#             s_jet4_boost.m.to_numpy()[:,np.newaxis], 
#             s_jet5_boost.m.to_numpy()[:,np.newaxis]))

#         xtra_excess_features = np.column_stack((
#             e_jet0_boost.pt.to_numpy()[:,np.newaxis], 
#             e_jet1_boost.pt.to_numpy()[:,np.newaxis], 
#             e_jet2_boost.pt.to_numpy()[:,np.newaxis], 
#             e_jet3_boost.pt.to_numpy()[:,np.newaxis], 
#             e_jet4_boost.pt.to_numpy()[:,np.newaxis], 
#             e_jet5_boost.pt.to_numpy()[:,np.newaxis],
#             e_jet0_boost.eta.to_numpy()[:,np.newaxis], 
#             e_jet1_boost.eta.to_numpy()[:,np.newaxis], 
#             e_jet2_boost.eta.to_numpy()[:,np.newaxis], 
#             e_jet3_boost.eta.to_numpy()[:,np.newaxis], 
#             e_jet4_boost.eta.to_numpy()[:,np.newaxis], 
#             e_jet5_boost.eta.to_numpy()[:,np.newaxis],
#             e_jet0_boost.phi.to_numpy()[:,np.newaxis], 
#             e_jet1_boost.phi.to_numpy()[:,np.newaxis], 
#             e_jet2_boost.phi.to_numpy()[:,np.newaxis], 
#             e_jet3_boost.phi.to_numpy()[:,np.newaxis], 
#             e_jet4_boost.phi.to_numpy()[:,np.newaxis], 
#             e_jet5_boost.m.to_numpy()[:,np.newaxis],
#             e_jet0_boost.m.to_numpy()[:,np.newaxis], 
#             e_jet1_boost.m.to_numpy()[:,np.newaxis], 
#             e_jet2_boost.m.to_numpy()[:,np.newaxis], 
#             e_jet3_boost.m.to_numpy()[:,np.newaxis], 
#             e_jet4_boost.m.to_numpy()[:,np.newaxis], 
#             e_jet5_boost.m.to_numpy()[:,np.newaxis]))

#         n_signal = len(xtra_signal_features)
#         n_excess = len(xtra_excess_features)

#         signal_targets = np.tile([1,0], (n_signal, 1))
#         excess_targets = np.tile([0,1], (n_excess, 1))

#         # n_signal = ak.count(tree.jet_idx[mixed_ind][excess_signal_mask], axis=1)
#         # excess_targets = np.zeros((nevents, 7), dtype=int)
#         # excess_targets[np.arange(nevents), n_signal] += 1

#         # signal_targets = np.zeros_like(excess_targets)
#         # signal_targets[:,-1] += 1

#         assert (np.any(~signal_targets[:,1]))
#         assert (np.all(signal_targets[:,0]))
#         assert (np.all(excess_targets[:,1]))
#         assert (~np.any(excess_targets[:,0]))

#         # Concatenate pT, eta, phi, and btag score
#         signal_features = np.concatenate((
#             signal_p4.pt.to_numpy(), 
#             signal_p4.pt.to_numpy()**2, 
#             signal_p4.eta.to_numpy(), 
#             signal_p4.eta.to_numpy()**2, 
#             signal_p4.phi.to_numpy(), 
#             signal_p4.phi.to_numpy()**2, 
#             signal_btag, 
#             xtra_signal_features), axis=1)
#         excess_features = np.concatenate((
#             excess_p4.pt.to_numpy(), 
#             excess_p4.pt.to_numpy()**2, 
#             excess_p4.eta.to_numpy(), 
#             excess_p4.eta.to_numpy()**2, 
#             excess_p4.phi.to_numpy(), 
#             excess_p4.phi.to_numpy()**2, 
#             excess_btag, 
#             xtra_excess_features), axis=1)

#         assert (signal_features.shape == excess_features.shape)

#         nan_mask = np.isnan(signal_features).any(axis=1)

#         features = np.row_stack((signal_features, excess_features))
#         good_mask = ~np.isnan(features).any(axis=1)
#         self.features = features[good_mask, :]
#         targets = np.row_stack((signal_targets, excess_targets))
#         self.targets = targets[good_mask, :]

#         assert len(self.features) == len(self.targets)
#         assert np.isnan(self.features).sum() == 0

#     def get_pair_p4(self, mask=None):
#         # if signal: mask = self.jet_idx > -1
#         # else: mask = self.jet_idx == -1
#         jet1_pt, jet2_pt = ak.unzip(ak.combinations(self.tree.jet_pt[mask], 2))
#         jet1_eta, jet2_eta = ak.unzip(ak.combinations(self.tree.jet_eta[mask], 2))
#         jet1_phi, jet2_phi = ak.unzip(ak.combinations(self.tree.jet_phi[mask], 2))
#         jet1_m, jet2_m = ak.unzip(ak.combinations(self.tree.jet_m[mask], 2))

#         jet1_p4 = vector.obj(
#             pt  = jet1_pt,
#             eta = jet1_eta,
#             phi = jet1_phi,
#             m   = jet1_m)

#         jet2_p4 = vector.obj(
#             pt  = jet2_pt,
#             eta = jet2_eta,
#             phi = jet2_phi,
#             m   = jet2_m)

#         return jet1_p4, jet2_p4

# class TrainTwo():

#     def __init__(self, filename):

#         tree = Tree(filename, 'sixBtree', as_ak=True)
#         nevents = len(tree.jet_pt)

#         n_sixb = tree.n_sixb
#         local_ind = ak.local_index(tree.jet_idx)
#         signal_jet_mask = tree.jet_idx > -1
#         signal_jet_ind  = local_ind[signal_jet_mask]
#         excess_jet_ind  = local_ind[~signal_jet_mask]

#         HX_mask, H1_mask, H2_mask = self.Higgs_masks(tree.jet_idx)
#         H_mask = HX_mask & H1_mask & H2_mask

#         mixed_ind = ak.sort(ak.concatenate((excess_jet_ind, signal_jet_ind), axis=1)[:, :6], axis=1)
#         mixed_ind_np = mixed_ind.to_numpy()

#         signal_p4 = vector.obj(
#             pt  = tree.jet_pt[signal_jet_mask],
#             eta = tree.jet_eta[signal_jet_mask],
#             phi = tree.jet_phi[signal_jet_mask],
#             m   = tree.jet_m[signal_jet_mask]
#         )

#         excess_p4 = vector.obj(
#             pt  = tree.jet_pt[mixed_ind],
#             eta = tree.jet_eta[mixed_ind],
#             phi = tree.jet_phi[mixed_ind],
#             m   = tree.jet_m[mixed_ind]
#         )

#         signal_btag = tree.jet_btag[signal_jet_mask].to_numpy()
#         excess_btag = tree.jet_btag[mixed_ind].to_numpy()

#         s_jet0_boosted_pt, s_jet1_boosted_pt, s_jet2_boosted_pt, s_jet3_boosted_pt, s_jet4_boosted_pt, s_jet5_boosted_pt = self.get_boosted(tree, signal_jet_mask)
#         e_jet0_boosted_pt, e_jet1_boosted_pt, e_jet2_boosted_pt, e_jet3_boosted_pt, e_jet4_boosted_pt, e_jet5_boosted_pt = self.get_boosted(tree, mixed_ind)

#         # Concatenate pT, eta, phi, and btag score
#         signal_inputs = np.concatenate((signal_p4.pt.to_numpy(), signal_p4.eta.to_numpy(), signal_p4.phi.to_numpy(), signal_btag, s_jet0_boosted_pt, s_jet1_boosted_pt, s_jet2_boosted_pt, s_jet3_boosted_pt, s_jet4_boosted_pt, s_jet5_boosted_pt), axis=1)
#         excess_inputs = np.concatenate((excess_p4.pt.to_numpy(), excess_p4.eta.to_numpy(), excess_p4.phi.to_numpy(), excess_btag, e_jet0_boosted_pt, e_jet1_boosted_pt, e_jet2_boosted_pt, e_jet3_boosted_pt, e_jet4_boosted_pt, e_jet5_boosted_pt), axis=1)
#         print(signal_inputs.shape)

#         self.signal_features = signal_inputs
#         self.excess_features = excess_inputs

#         def Higgs_masks(self, jet_idx):

#             mask0 = ak.where(jet_idx == 0, 1, 0) # find HX b1
#             mask1 = ak.where(jet_idx == 1, 1, 0) # find HX b1
#             HX_mask = ak.where(mask1, 1, mask0) # include HX b2

#             mask2 = ak.where(jet_idx == 2, 1, 0) # find HY1 b1
#             mask3 = ak.where(jet_idx == 3, 1, 0) # find HY1 b1
#             H1_mask = ak.where(mask3, 1, mask2) # include HY1 b2

#             mask4 = ak.where(jet_idx == 4, 1, 0)# find HY2 b1
#             mask5 = ak.where(jet_idx == 5, 1, 0)# find HY2 b1
#             H2_mask = ak.where(mask5, 1, mask4)# include HY2 b2

#         return  HX_mask, H1_mask, H2_mask

# class combos():

#     def __init__(self, filename, njets=7, tag6b=None, tag2b=None):

#         self.evaluate_6b(filename, njets, tag6b)
#         # self.evaluate_2b(filename, njets, tag2b)``

#     def evaluate_2b(self, filename, njets, tag):
#         print("Loading...")
#         tree = Tree(filename, 'sixBtree', as_ak=True)
#         nevents = len(tree.jet_pt)

#         # Arrays of indices representing the pt-ordered event
#         jet_ptordered = ak.argsort(tree.jet_pt, ascending=False)
#         # Arrays containing indices representing all 6-jet combinations in each event
#         jet_comb = ak.combinations(jet_ptordered, 2)
#         # Unzip the combinations into their constituent jets
#         jet0, jet1 = ak.unzip(jet_comb)
#         # Zip the constituents back together
#         combos = ak.concatenate([jeti[:,:,np.newaxis] for jeti in (jet0, jet1)], axis=-1)

#         print("Broadcasting...")
#         pt1, pt2 = ak.unzip(ak.combinations(tree.jet_pt, njets))
#         eta1, eta2 = ak.unzip(ak.combinations(tree.jet_eta, njets))
#         phi1, phi2 = ak.unzip(ak.combinations(tree.jet_phi, njets))
#         m1, m2 = ak.unzip(ak.combinations(tree.jet_m, njets))
#         btag1, btag2 = ak.unzip(ak.combinations(tree.jet_btag, njets))
#         idx1, idx2 = ak.unzip(ak.combinations(tree.jet_idx, njets))

#         # jets = [jet0, jet1]
#         # [jet0_boosted_pt, jet1_boosted_pt] = self.get_boosted_p4(tree, jets)
#         jet0_p4 = p4(tree, jet0)
#         jet1_p4 = p4(tree, jet1)
#         jet_ids = combos.to_numpy() # (nevents, ncombos, 2) array of jet ids in a pair
#         deltaR = ak.flatten(jet0_p4.deltaR(jet1_p4)).to_numpy()

#         print("Concatenating inputs...")
#         # Concatenate pT, eta, phi, and btag score
#         inputs = np.concatenate((pt, eta, phi, btag), axis=2)
#         D1 = inputs.shape[0]
#         D2 = inputs.shape[1]
#         D3 = inputs.shape[2]

#         print("Defining features...")
#         # Reshape inputs to match the structure of the boosted pTs and concatenate all of it together
#         features = np.column_stack((inputs.reshape(D1*D2, D3), deltaR))

#         print("Labeling signal...")
#         # jet_idx provides the matching index (-1 if unmatched) to gen jets
#         idx_shape = idx[combos].shape
#         # Signal mask for combinations
#         combo_idx = idx.reshape(idx_shape[0]*idx_shape[1], idx_shape[2])
#         HX_mask = np.all(np.logical_or(combo_idx == 0, combo_idx == 1), axis=1)
#         H1_mask = np.all(np.logical_or(combo_idx == 2, combo_idx == 3), axis=1)
#         H2_mask = np.all(np.logical_or(combo_idx == 4, combo_idx == 5), axis=1)
#         Higgs_mask = np.logical_or(np.logical_or(HX_mask, H1_mask), H2_mask)
#         evt_mask = Higgs_mask.reshape(nevents, ncombos)
#         sgnl_ind = np.argwhere(Higgs_mask) # (length = nevents where sgnl is found)

#         print("Loading model...")
#         # Load scaler and model
#         scaler, model = load_model(location='../../2jet_classifier/', tag=tag)
#         # Generate predictions by applying the model to the features
#         scores = model.predict(scaler.transform(features))[:,0]

#         # Change into shape of events containing 7 distinct combinations
#         scores_reshaped = scores.reshape(nevents, ncombos)

#         # Return maximum score found in the event
#         max_evt_mask = np.argsort(scores_reshaped, axis=1)[:,::-1,np.newaxis]
#         jet_ids_by_score = np.take_along_axis(jet_ids, max_evt_mask, 1)
#         max_evt_score = np.sort(scores_reshaped, axis=1)[:,::-1]

#         self.obtain_unique_pairs(jet_ids_by_score)

#         sgnl_is_max = np.equal(sgnl_ind, max_evt_mask[evt_mask])
#         print(sgnl_is_max.sum()/len(sgnl_is_max))

#     def obtain_unique_pairs(self, sorted_jet_ids):
#         distinct_pairs = np.ones((sorted_jet_ids.shape[0], 3)) * -1
#         print(distinct_pairs)
#         sys.exit()
#         for _ in range(sorted_jet_ids.shape[1]):
#             sorted_jet_ids

#     def evaluate_6b(self, filename, njets, tag):

#         print("Initializing...")

#         tree = Tree(filename, 'sixBtree', training=True)
#         nevents = len(tree.jet_pt)

#         print("Indexing...")

#         jet1pt, jet2pt, jet3pt, jet4pt, jet5pt, jet6pt = ak.unzip(ak.combinations(tree.jet_pt, njets))
#         sys.exit()




#         # Arrays of indices representing the pt-ordered event
#         # jet_ptordered = ak.argsort(tree.jet_pt, ascending=False)
#         # Arrays containing indices representing all 6-jet combinations in each event
#         jet_index = ak.local_index(tree.jet_pt)
#         jet_comb = ak.combinations(jet_index, njets)
#         # Unzip the combinations into their constituent jets
#         jet0, jet1, jet2, jet3, jet4, jet5 = ak.unzip(jet_comb)
#         # Zip the constituents back together
#         combos = ak.concatenate([jeti[:,:,np.newaxis] for jeti in (jet0, jet1, jet2, jet3, jet4, jet5)], axis=-1)

#         pt1, pt2 = ak.unzip(ak.combinations(tree.jet_pt, njets))
#         eta1, eta2 = ak.unzip(ak.combinations(tree.jet_eta, njets))
#         phi1, phi2 = ak.unzip(ak.combinations(tree.jet_phi, njets))
#         m1, m2 = ak.unzip(ak.combinations(tree.jet_m, njets))
#         btag1, btag2 = ak.unzip(ak.combinations(tree.jet_btag, njets))
#         idx1, idx2 = ak.unzip(ak.combinations(tree.jet_idx, njets))

#         signal_mask = idx > -1
#         n_signal = ak.count(idx[signal_mask], axis=2)
#         n_signal = n_signal.to_numpy()
#         self.n_signal = n_signal.reshape(n_signal.shape[0]*n_signal.shape[1])
#         idx = idx.to_numpy()

#         jets = [jet0, jet1, jet2, jet3, jet4, jet5]
#         [jet0_boosted_pt, jet1_boosted_pt, jet2_boosted_pt, jet3_boosted_pt, jet4_boosted_pt, jet5_boosted_pt] = get_boosted_p4(tree, jets)

#         # Concatenate pT, eta, phi, and btag score
#         inputs = np.concatenate((pt, eta, phi, btag), axis=2)
#         D1 = inputs.shape[0]
#         D2 = inputs.shape[1]
#         D3 = inputs.shape[2]

#         # Reshape inputs to match the structure of the boosted pTs and concatenate all of it together
#         features = np.concatenate((inputs.reshape(D1*D2, D3), jet0_boosted_pt, jet1_boosted_pt, jet2_boosted_pt, jet3_boosted_pt, jet4_boosted_pt, jet5_boosted_pt), axis=1)
        
#         # jet_idx provides the matching index (-1 if unmatched) to gen jets
#         idx_shape = ak.to_numpy(idx).shape
#         # Signal mask for combinations
#         signal_mask = np.all(ak.to_numpy(idx).reshape(idx_shape[0]*idx_shape[1], idx_shape[2]) > -1, axis=1)
#         evt_mask = signal_mask.reshape(nevents, njets)
#         evt_sgnl_mask = np.any(evt_mask, axis=1)
#         sgnl_ind = np.argwhere(evt_mask)[:,1] # (length = nevents where sgnl is found)

#         # Load scaler and model
#         scaler, model = load_model(location='../', tag=tag)
#         # Generate predictions by applying the model to the features
#         scores = model.predict(scaler.transform(features))
#         self.scores = scores
#         self.category = np.argmax(scores, axis=1)

#         # Change into shape of events containing 7 distinct combinations
#         # scores_reshaped = scores.reshape(nevents, njets)

#         # Return maximum score found in the event
#         # max_evt_mask = scores_reshaped.argmax(axis=1)
#         # max_evt_score = scores_reshaped.max(axis=1)

#         # sgnl_is_max = np.equal(sgnl_ind, max_evt_mask[evt_sgnl_mask])
#         # print(sgnl_is_max.sum()/len(sgnl_is_max))

        

#     def apply_6j_model(self, features, tag, location='../'):

#         print("Applying 6b Model to combinations. Please wait.")
#         scaler, model = load_model(location, tag)
#         test_features = scaler.transform(combo_features)
#         scores_combo = model.predict(test_features)[:,0]

#         print("Selecting highest scoring combination from each event.")
#         # self.select_highest_scoring_combos()


