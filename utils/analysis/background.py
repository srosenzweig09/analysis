""" 
Author: Suzanne Rosenzweig
"""

from utils import *
from utils.varUtils import *
from utils.plotter import latexTitle
from utils.analysis.particle import Particle

# Standard library imports
import sys 
import uproot
import subprocess, shlex

import awkward0 as ak0

vector.register_awkward()

current_model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/XY_3H_reco_ranker/20230213_ranger_lr0.0047_batch1024__full_reco_withbkg/predict_output'
current_model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/XY_3H_reco_ranker/20230215_ranger_lr0.0047_batch1024__withbkg/predict_output'

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

class Bkg():
    
    def __init__(self, filename, treename='sixBtree', year=2018, training=False, exploration=False, signal=True, gnn_model=None):
        """
        A class for handling TTrees from older skims, which output an array of jet kinematics. (Newer skims output a single branch for each b jet kinematic.)

        args:
            filename: string containing name of a single file OR a list of several files to open
            treename: default is 'sixBtree,' which is the named TTree output from the analysis code

        returns:
            Nothing. Initializes attributes of Tree class.
        """

        if type(filename) != list:
            self.single_init(filename, treename, year, gnn_model=gnn_model)
        else:
            self.multi_init(filename, treename, year, gnn_model=gnn_model)

    def single_init(self, filename, treename, year, gnn_model):
        """Opens a single file into a TTree"""
        tree = uproot.open(f"{filename}:{treename}")
        self.tree = tree

        # if not exploration:
        #     for k, v in tree.items():
        #         if ('t6' in k) or ('score' in k) or ('nn' in k): 
        #             setattr(self, k, v.array())
        #             used_key = k
        # else:            
        for k, v in tree.items():
            if (k.startswith('jet_') or (k.startswith('H_'))):
                setattr(self, k, v.array())
                used_key = k

        self.nevents = len(tree[used_key].array())

        cutflow = uproot.open(f"{filename}:h_cutflow_unweighted")
        # save total number of events for scaling purposes
        total = cutflow.to_numpy()[0][0]
        _, xsec = next( ((key,value) for key,value in xsecMap.items() if key in filename),("unk",1) )
        # if signal: 
        #     self.sample = latexTitle(filename)
        #     self.mXmY = self.sample.replace('$','').replace('_','').replace('= ','_').replace(', ','_').replace(' GeV','')
        self.xsec = xsec
        self.lumi = lumiMap[year][0]
        self.scale = self.lumi*xsec/total
        self.cutflow = cutflow.to_numpy()[0]*self.scale

    def multi_init(self, filelist, treename, year, gnn_model):
        """Opens a list of files into several TTrees. Typically used for bkgd events."""
        self.is_signal = False
        self.is_bkgd = True

        arr_trees = []
        arr_xsecs = []
        arr_nevent = []
        arr_sample = []
        arr_total = []
        self.arr_n = []
        self.maxcomb = []

        

        self.cutflow = np.zeros(11)

        predictions = subprocess.check_output(shlex.split(f"ls {current_model_path}"))
        predictions = predictions.decode('UTF-8').split('\n')[:-1]

        for filename in filelist:
            # Open tree
            # tree = uproot.open(f"{filename}:{treename}")
            with uproot.open(f"{filename}:{treename}") as tree:
                # How many events in the tree?
                n = len(tree['jet_pt'].array())
                self.arr_n.append(n)
                cutflow = uproot.open(f"{filename}:h_cutflow")
                # save total number of events for scaling purposes
                samp_tot = cutflow.to_numpy()[0][0]
                cf = cutflow.to_numpy()[0]
                if len(cf) < 11: cf = np.append(cf, np.zeros(11-len(cf)))
                if n == 0: continue # Skip if no events
                arr_total.append(samp_tot)
                arr_nevent.append(n)
                # Bkgd events must be scaled to the appropriate xs in order to compare fairly with signal yield
                samp, xsec = next( ((key,value) for key,value in xsecMap.items() if key in filename),("unk",1) )
                print(samp)
                samp_file = f"{current_model_path}/{[i for i in predictions if samp in i][0]}"
                print(samp_file)
                with ak0.load(samp_file) as f_awk:
                    # self.scores = ak.unflatten(f_awk['scores'], np.repeat(45, n)).to_numpy()
                    # self.maxscore = f_awk['maxscore']
                    # combos = ak.from_numpy(f_awk['maxcomb'])
                    combos = ak.from_regular(f_awk['maxcomb'])
                    # combos = ak.unflatten(ak.flatten(combos), ak.ones_like(combos[:,0])*6)
                    self.maxcomb.append(combos)
                    
                    # self.maxlabel = f_awk['maxlabel']

                self.cutflow += cf[:11] * lumiMap[year][0] * xsec / samp_tot
                arr_trees.append(tree)
                arr_xsecs.append(xsec)
                arr_sample.append(samp)
                setattr(self, samp, tree)
                for k, v in tree.items():
                    setattr(getattr(self, samp), k, v.array())

        self.ntrees = len(arr_trees)
        self.tree = arr_trees
        self.xsec = arr_xsecs
        self.lumi = lumiMap[year][0]
        self.nevents = arr_nevent
        self.sample = arr_sample
        self.total = arr_total
        self.scale = self.lumi*np.asarray(arr_xsecs)/np.asarray(arr_total)
        self.scale = np.repeat(self.scale, np.array(self.arr_n))
        # self.weighted_n = np.asarray(self.nevents)*self.scale

        for k in tree.keys():
            if 'jet' in k:
                leaves = []
                for samp in self.sample:
                    arr = getattr(getattr(self, samp), k)
                    leaves.append(arr)
                setattr(self, k, leaves)

        if gnn_model is not None: self.init_from_gnn(gnn_model)

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
            return n#, b, centers
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
        return n*self.scale#, b, centers

    def hist(self, key):
        # print(getattr(self, key))
        scale = np.repeat(self.scale, np.array(self.arr_n))
        return ak.concatenate(getattr(self, key))

    def keys(self):
        print(self.tree.keys())

    def get(self, key):
        return self.tree[key].array()

    def init_from_gnn(self):
        pt = ak.concatenate([pt[comb] for pt,comb in zip(self.jet_ptRegressed,self.maxcomb)])
        eta = ak.concatenate([eta[comb] for eta,comb in zip(self.jet_eta,self.maxcomb)])
        phi = ak.concatenate([phi[comb] for phi,comb in zip(self.jet_phi,self.maxcomb)])
        m = ak.concatenate([m[comb] for m,comb in zip(self.jet_mRegressed,self.maxcomb)])
        btag = ak.concatenate([btag[comb] for btag,comb in zip(self.jet_btag,self.maxcomb)])

        HX_b1 = Particle(kin_dict={'pt':pt[:,0],'eta':eta[:,0],'phi':phi[:,0],'m':m[:,0],'btag':btag[:,0]})
        HX_b2 = Particle(kin_dict={'pt':pt[:,1],'eta':eta[:,1],'phi':phi[:,1],'m':m[:,1],'btag':btag[:,1]})
        H1_b1 = Particle(kin_dict={'pt':pt[:,2],'eta':eta[:,2],'phi':phi[:,2],'m':m[:,2],'btag':btag[:,2]})
        H1_b2 = Particle(kin_dict={'pt':pt[:,3],'eta':eta[:,3],'phi':phi[:,3],'m':m[:,3],'btag':btag[:,3]})
        H2_b1 = Particle(kin_dict={'pt':pt[:,4],'eta':eta[:,4],'phi':phi[:,4],'m':m[:,4],'btag':btag[:,4]})
        H2_b2 = Particle(kin_dict={'pt':pt[:,5],'eta':eta[:,5],'phi':phi[:,5],'m':m[:,5],'btag':btag[:,5]})

        self.HX = HX_b1 + HX_b2
        self.H1 = H1_b1 + H1_b2
        self.H2 = H2_b1 + H2_b2

        self.Y = self.H1 + self.H2