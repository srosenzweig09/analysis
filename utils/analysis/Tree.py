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
from .varUtils import *
from .plotter import latexTitle

# Standard library imports
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

class Tree():
    
    def __init__(self, filename, treename='sixBtree', year=2018, training=False, exploration=False, signal=True):
        """
        A class for handling TTrees from older skims, which output an array of jet kinematics. (Newer skims output a single branch for each b jet kinematic.)

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

    def keys(self):
        print(self.tree.keys())

    def get(self, key):
        return self.tree[key].array()