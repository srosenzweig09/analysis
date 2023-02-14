""" 
Author: Suzanne Rosenzweig
"""

from utils import *
from utils.varUtils import *
from utils.plotter import latexTitle

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
            n = len(tree['jet_pt'].array())
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

    # def hist(self):


    def keys(self):
        print(self.tree.keys())

    def get(self, key):
        return self.tree[key].array()

    def init_from_gnn(self):
        