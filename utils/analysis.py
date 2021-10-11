"""
Project: 6b Final States - 
Author: Suzanne Rosenzweig

Classes:
Tree - Extracts information from ROOT TTrees.
TrainSix - Produces inputs to train the 6-jet classifier.


Notes:
Training samples are prepared such that these requirements are already imposed:
- n_jet > 6
- n_sixb == 6
"""

from . import *
from .plotter import easy_bins
from .modelUtils.load import load_model

# Standard library imports
from math import comb
import sys 
import uproot

vector.register_awkward()

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

class Tree():
    def __init__(self, filename, treename='sixBtree', year=2018, training=False):
        """
        A class for handling TTrees.

        args:
            filename: string containing name of a single file OR a list of several files to open
            treename: default is 'sixBtree,' which is the named TTree output from the analysis code

        returns:
            Nothing. Initializes attributes of Tree class.
        """

        if type(filename) != list:
            self.single_init(filename, treename, year, training=training)
        else:
            self.multi_init(filename, treename, year)

    def single_init(self, filename, treename, year=2018, training=False):
        """Opens a single file into a TTree"""
        self.is_signal = True
        self.is_bkgd = False
        tree = uproot.open(f"{filename}:{treename}")
        self.tree = tree
        for k, v in tree.items():
            setattr(self, k, v.array())
        nevents = len(tree[k].array())

        self.local_ind = ak.local_index(self.jet_pt)

        try:
            self.signal_evt_mask = ak.sum(self.jet_signalId > -1, axis=1) == 6
            self.signal_jet_mask = self.jet_signalId[self.signal_evt_mask] > -1
        except:
            self.signal_evt_mask = ak.sum(self.jet_idx > -1, axis=1) == 6
            self.signal_jet_mask = self.jet_idx[self.signal_evt_mask] > -1

        if training: return
        cutflow = uproot.open(f"{filename}:h_cutflow")
        # save total number of events for scaling purposes
        total = cutflow.to_numpy()[0][0]
        samp, xsec = next( ((key,value) for key,value in xsecMap.items() if key in filename),("unk",1) )
        self.sample = samp
        self.xsec = xsec
        self.lumi = lumiMap[year][0]
        self.scale = self.lumi*xsec/total
        # self.weight = self.scale*nevents
        self.cutflow = cutflow.to_numpy()[0]*self.scale

        for k, v in tree.items():
            try: setattr(self, 'signal_' + k, v.array()[self.signal_evt_mask][self.signal_jet_mask])
            except: continue

    def multi_init(self, filelist, treename, year=2018):
        """Opens a list of files into several TTrees. Typically used for bkgd events."""
        self.is_signal = False
        self.is_bkgd = True
        trees = []
        xsecs = []
        nevent = []
        sample = []
        total = []
        cutflow = np.zeros(11)
        for filename in filelist:
            # Open tree
            tree = uproot.open(f"{filename}:{treename}")
            # How many events in the tree?
            n = len(tree['n_jet'].array())
            if n == 0: continue # Skip if no events
            cutflow = uproot.open(f"{filename}:h_cutflow")
            # save total number of events for scaling purposes
            total.append(cutflow.to_numpy()[0][0])
            nevent.append(n)
            trees.append(tree)
            # Bkgd events must be scaled to the appropriate xs in order to compare fairly with signal yield
            samp, xsec = next( ((key,value) for key,value in xsecMap.items() if key in filename),("unk",1) )
            sample.append(samp)
            xsecs.append(xsec)
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

    def get_branches(self, key):
        for sample in self.sample:
            
            n, b = np.histogram()
        return branches

    def get_hist_weights(self, key, **kwargs):
        for k,v in easy_bins.items():
            if k in key: bins = v
        if self.is_bkgd: 
            n = []
            for sample, scale in zip(self.sample, self.scale):
                branch = ak.flatten(getattr(getattr(self, sample), key)).to_numpy()
                n_i, b = np.histogram(branch, bins, **kwargs)
                centers = (b[1:] + b[:-1])/2
                n.append(n_i*scale)
            n = np.asarray(n)
            n = np.sum(n, axis=0)
            return n, b, centers
        else: 
            branch = ak.flatten(getattr(self, key)).to_numpy()
            n, b = np.histogram(branch, bins, **kwargs)
            centers = (b[1:] + b[:-1])/2
            # n_tot = sum(n)
            # epsilon = 5e-3
            # max_bin = sum(n > epsilon*n_tot) + 1
            # new_bins = np.linspace(branch.min(), bins[max_bin], 100)
            # n, b = np.histogram(branch, new_bins, **kwargs)
            # centers = (b[1:] + b[:-1])/2
        return n*self.scale, b, centers

    def get_pair_p4(self, mask=None):
        # if signal: mask = self.jet_idx > -1
        # else: mask = self.jet_idx == -1
        jet1_pt, jet2_pt = ak.unzip(ak.combinations(self.jet_pt[mask], 2))
        jet1_eta, jet2_eta = ak.unzip(ak.combinations(self.jet_eta[mask], 2))
        jet1_phi, jet2_phi = ak.unzip(ak.combinations(self.jet_phi[mask], 2))
        jet1_m, jet2_m = ak.unzip(ak.combinations(self.jet_m[mask], 2))

        jet1_p4 = vector.obj(
            pt  = jet1_pt,
            eta = jet1_eta,
            phi = jet1_phi,
            m   = jet1_m)

        jet2_p4 = vector.obj(
            pt  = jet2_pt,
            eta = jet2_eta,
            phi = jet2_phi,
            m   = jet2_m)

        return jet1_p4, jet2_p4

    def get_t6_pair_p4(self):
        # if signal: mask = self.jet_idx > -1
        # else: mask = self.jet_idx == -1
        jet1_pt, jet2_pt = ak.unzip(ak.combinations(self.jet_pt[:,:6], 2))
        jet1_eta, jet2_eta = ak.unzip(ak.combinations(self.jet_eta[:,:6], 2))
        jet1_phi, jet2_phi = ak.unzip(ak.combinations(self.jet_phi[:,:6], 2))
        jet1_m, jet2_m = ak.unzip(ak.combinations(self.jet_m[:,:6], 2))
        jet1_idx, jet2_idx = ak.unzip(ak.combinations(self.jet_signalId[:,:6], 2))
        jet12_ind = ak.combinations(ak.local_index(self.jet_pt[:,:6]),2)
        jet1_ind, jet2_ind = ak.unzip(jet12_ind)

        jet1_p4 = vector.obj(
            pt  = jet1_pt,
            eta = jet1_eta,
            phi = jet1_phi,
            m   = jet1_m)

        jet2_p4 = vector.obj(
            pt  = jet2_pt,
            eta = jet2_eta,
            phi = jet2_phi,
            m   = jet2_m)

        return jet1_p4, jet2_p4, jet1_ind, jet2_ind, jet12_ind, jet1_idx, jet2_idx
    
    def p4(self, pt, eta, phi, m):
        return vector.obj(pt=pt, eta=eta, phi=phi, m=m)

    def get_all_p4(self, pts, etas, phis, ms):
        return [self.p4(pt, eta, phi, m) for pt,eta,phi,m in zip(pts,etas,phis,ms)]

    def get_six_p4(self, mask=None):
        jet_pt = ak.unzip(ak.combinations(self.jet_pt[mask], 6))
        jet_eta = ak.unzip(ak.combinations(self.jet_eta[mask], 6))
        jet_phi = ak.unzip(ak.combinations(self.jet_phi[mask], 6))
        jet_m = ak.unzip(ak.combinations(self.jet_m[mask], 6))

        return self.get_all_p4(jet_pt, jet_eta, jet_phi, jet_m)

    def sort(self, mask):
        for k, v in self.tree.items():
            try: setattr(self, k, v.array()[mask])
            except: continue

    def keys(self):
        print(self.tree.keys())

    def get(self, key):
        return self.tree[key]

    def get_combos(self, n, signal=False):
        """
        Return a tree mask that produces all possible combinations of n jets from each event.
        """
        if signal: mask = self.local_ind[self.signal_evt_mask][self.signal_jet_mask]
        else: mask = ak.local_index(self.jet_pt)
        # Arrays of indices representing the pt-ordered event
        local_index = ak.local_index(self.jet_pt[mask])
        # Arrays containing indices representing all 6-jet combinations in each event
        jet_comb = ak.combinations(local_index, n)
        # Unzip the combinations into their constituent jets
        jets = ak.unzip(jet_comb)
        # Zip the constituents back together
        self.combos = ak.concatenate([jeti[:,:,np.newaxis] for jeti in jets], axis=-1)

    def get_2j_input(self):
        pt1, pt2 = ak.unzip(ak.combinations(self.jet_pt, 2))
        eta1, eta2 = ak.unzip(ak.combinations(self.jet_eta, 2))
        phi1, phi2 = ak.unzip(ak.combinations(self.jet_phi, 2))
        m1, m2 = ak.unzip(ak.combinations(self.jet_m, 2))
        btag1, btag2 = ak.unzip(ak.combinations(self.jet_btag, 2))
        sid1, sid2 = ak.unzip(ak.combinations(self.jet_signalId, 2))

        jet1 = vector.obj(pt=pt1, eta=eta1, phi=phi1, m=m1)
        jet2 = vector.obj(pt=pt2, eta=eta2, phi=phi2, m=m2)

        dijet_pt = (jet1 + jet2).pt
        deltaR = jet1.deltaR(jet2)

        dijet_mass = (jet1 + jet2).m

        n_evt = ak.count(pt1, axis=1)


        pt1 = ak.flatten(pt1).to_numpy()
        pt2 = ak.flatten(pt2).to_numpy()
        eta1 = ak.flatten(eta1).to_numpy()
        eta2 = ak.flatten(eta2).to_numpy()
        phi1 = ak.flatten(phi1).to_numpy()
        phi2 = ak.flatten(phi2).to_numpy()
        btag1 = ak.flatten(btag1).to_numpy()
        btag2 = ak.flatten(btag2).to_numpy()
        sid1 = ak.flatten(sid1).to_numpy()
        sid2 = ak.flatten(sid2).to_numpy()
        HX_mask = ((sid1 == 0) & (sid2 == 1)) | ((sid2 == 0) & (sid1 == 1))
        H1_mask = ((sid1 == 2) & (sid2 == 3)) | ((sid2 == 2) & (sid1 == 3))
        H2_mask = ((sid1 == 4) & (sid2 == 5)) | ((sid2 == 4) & (sid1 == 5))
        dijet_pt = ak.flatten(dijet_pt).to_numpy()
        deltaR = ak.flatten(deltaR).to_numpy()


        signal_mask = HX_mask | H1_mask | H2_mask

        inputs = np.column_stack((pt1, pt2, eta1, eta2, phi1, phi2, btag1, btag2, dijet_pt, deltaR))

        scaler, model = load_model(location='../2jet_classifier/', tag='20210817_4btag_req')
        # scaled_inputs = scaler.transform(inputs[:,:-1])
        # print(scaled_inputs[:,-1])
        scaled_inputs = scaler.transform(np.column_stack((inputs[:,:-2], inputs[:,-1])))
        # print(scaled_inputs[:,-1])
        scores = model.predict(scaled_inputs)

        return scores, dijet_mass, signal_mask, n_evt

class TrainSix():
    
    def get_boosted(self, tree, ind_array):

        jet0_p4 = build_p4(tree.jet_pt[ind_array][:,0], 
                           tree.jet_eta[ind_array][:,0], 
                           tree.jet_phi[ind_array][:,0], 
                           tree.jet_m[ind_array][:,0])
        jet1_p4 = build_p4(tree.jet_pt[ind_array][:,1], 
                           tree.jet_eta[ind_array][:,1], 
                           tree.jet_phi[ind_array][:,1], 
                           tree.jet_m[ind_array][:,1])
        jet2_p4 = build_p4(tree.jet_pt[ind_array][:,2], 
                           tree.jet_eta[ind_array][:,2], 
                           tree.jet_phi[ind_array][:,2], 
                           tree.jet_m[ind_array][:,2])
        jet3_p4 = build_p4(tree.jet_pt[ind_array][:,3], 
                           tree.jet_eta[ind_array][:,3], 
                           tree.jet_phi[ind_array][:,3], 
                           tree.jet_m[ind_array][:,3])
        jet4_p4 = build_p4(tree.jet_pt[ind_array][:,4], 
                           tree.jet_eta[ind_array][:,4], 
                           tree.jet_phi[ind_array][:,4], 
                           tree.jet_m[ind_array][:,4])
        jet5_p4 = build_p4(tree.jet_pt[ind_array][:,5], 
                           tree.jet_eta[ind_array][:,5], 
                           tree.jet_phi[ind_array][:,5], 
                           tree.jet_m[ind_array][:,5])

        jet6_p4 = jet0_p4 + jet1_p4 + jet2_p4 + jet3_p4 + jet4_p4 + jet5_p4

        jet0_boost = jet0_p4.boost_p4(jet6_p4)
        jet1_boost = jet1_p4.boost_p4(jet6_p4)
        jet2_boost = jet2_p4.boost_p4(jet6_p4)
        jet3_boost = jet3_p4.boost_p4(jet6_p4)
        jet4_boost = jet4_p4.boost_p4(jet6_p4)
        jet5_boost = jet5_p4.boost_p4(jet6_p4)

        # jet0_boosted_pt = jet0_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        # jet1_boosted_pt = jet1_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        # jet2_boosted_pt = jet2_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        # jet3_boosted_pt = jet3_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        # jet4_boosted_pt = jet4_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        # jet5_boosted_pt = jet5_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]

        return jet0_boost, jet1_boost, jet2_boost, jet3_boost, jet4_boost, jet5_boost, jet6_p4

    def get_p4s(self, tree, ind_array):

        jet0_p4 = build_p4(tree.jet_pt[ind_array][:,0], 
                           tree.jet_eta[ind_array][:,0], 
                           tree.jet_phi[ind_array][:,0], 
                           tree.jet_m[ind_array][:,0])
        jet1_p4 = build_p4(tree.jet_pt[ind_array][:,1], 
                           tree.jet_eta[ind_array][:,1], 
                           tree.jet_phi[ind_array][:,1], 
                           tree.jet_m[ind_array][:,1])
        jet2_p4 = build_p4(tree.jet_pt[ind_array][:,2], 
                           tree.jet_eta[ind_array][:,2], 
                           tree.jet_phi[ind_array][:,2], 
                           tree.jet_m[ind_array][:,2])
        jet3_p4 = build_p4(tree.jet_pt[ind_array][:,3], 
                           tree.jet_eta[ind_array][:,3], 
                           tree.jet_phi[ind_array][:,3], 
                           tree.jet_m[ind_array][:,3])
        jet4_p4 = build_p4(tree.jet_pt[ind_array][:,4], 
                           tree.jet_eta[ind_array][:,4], 
                           tree.jet_phi[ind_array][:,4], 
                           tree.jet_m[ind_array][:,4])
        jet5_p4 = build_p4(tree.jet_pt[ind_array][:,5], 
                           tree.jet_eta[ind_array][:,5], 
                           tree.jet_phi[ind_array][:,5], 
                           tree.jet_m[ind_array][:,5])

        jet6_p4 = jet0_p4 + jet1_p4 + jet2_p4 + jet3_p4 + jet4_p4 + jet5_p4

        return jet0_p4, jet1_p4, jet2_p4, jet3_p4, jet4_p4, jet5_p4

    def __init__(self, filename, dijet=False):

        tree = Tree(filename, 'sixBtree', training=True)
        self.tree = tree
        nevents = len(tree.jet_pt)

        n_sixb = tree.n_sixb
        local_ind = ak.local_index(tree.jet_idx)
        signal_jet_mask = tree.jet_idx > -1
        signal_jet_ind  = local_ind[signal_jet_mask]
        excess_jet_ind  = local_ind[~signal_jet_mask]
        mixed_ind = ak.sort(ak.concatenate((excess_jet_ind, signal_jet_ind), axis=1)[:, :6], axis=1)
        mixed_ind_np = mixed_ind.to_numpy()
        excess_signal_mask = tree.jet_idx[mixed_ind] > -1

        signal_p4 = vector.obj(
            pt  = tree.jet_pt[signal_jet_ind],
            eta = tree.jet_eta[signal_jet_ind],
            phi = tree.jet_phi[signal_jet_ind],
            m   = tree.jet_m[signal_jet_ind]
        )

        excess_p4 = vector.obj(
            pt  = tree.jet_pt[mixed_ind],
            eta = tree.jet_eta[mixed_ind],
            phi = tree.jet_phi[mixed_ind],
            m   = tree.jet_m[mixed_ind]
        )

        signal_btag = tree.jet_btag[signal_jet_ind].to_numpy()
        excess_btag = tree.jet_btag[mixed_ind].to_numpy()


        s_jet0_boost, s_jet1_boost, s_jet2_boost, s_jet3_boost, s_jet4_boost, s_jet5_boost, s_jet6_p4 = self.get_boosted(tree, signal_jet_mask)
        e_jet0_boost, e_jet1_boost, e_jet2_boost, e_jet3_boost, e_jet4_boost, e_jet5_boost, e_jet6_p4 = self.get_boosted(tree, mixed_ind)

        xtra_signal_features = np.column_stack((
            s_jet0_boost.pt.to_numpy()[:,np.newaxis], 
            s_jet1_boost.pt.to_numpy()[:,np.newaxis], 
            s_jet2_boost.pt.to_numpy()[:,np.newaxis], 
            s_jet3_boost.pt.to_numpy()[:,np.newaxis], 
            s_jet4_boost.pt.to_numpy()[:,np.newaxis], 
            s_jet5_boost.pt.to_numpy()[:,np.newaxis],
            s_jet0_boost.eta.to_numpy()[:,np.newaxis], 
            s_jet1_boost.eta.to_numpy()[:,np.newaxis], 
            s_jet2_boost.eta.to_numpy()[:,np.newaxis], 
            s_jet3_boost.eta.to_numpy()[:,np.newaxis], 
            s_jet4_boost.eta.to_numpy()[:,np.newaxis], 
            s_jet5_boost.eta.to_numpy()[:,np.newaxis],
            s_jet0_boost.phi.to_numpy()[:,np.newaxis], 
            s_jet1_boost.phi.to_numpy()[:,np.newaxis], 
            s_jet2_boost.phi.to_numpy()[:,np.newaxis], 
            s_jet3_boost.phi.to_numpy()[:,np.newaxis], 
            s_jet4_boost.phi.to_numpy()[:,np.newaxis], 
            s_jet5_boost.m.to_numpy()[:,np.newaxis],
            s_jet0_boost.m.to_numpy()[:,np.newaxis], 
            s_jet1_boost.m.to_numpy()[:,np.newaxis], 
            s_jet2_boost.m.to_numpy()[:,np.newaxis], 
            s_jet3_boost.m.to_numpy()[:,np.newaxis], 
            s_jet4_boost.m.to_numpy()[:,np.newaxis], 
            s_jet5_boost.m.to_numpy()[:,np.newaxis]))

        xtra_excess_features = np.column_stack((
            e_jet0_boost.pt.to_numpy()[:,np.newaxis], 
            e_jet1_boost.pt.to_numpy()[:,np.newaxis], 
            e_jet2_boost.pt.to_numpy()[:,np.newaxis], 
            e_jet3_boost.pt.to_numpy()[:,np.newaxis], 
            e_jet4_boost.pt.to_numpy()[:,np.newaxis], 
            e_jet5_boost.pt.to_numpy()[:,np.newaxis],
            e_jet0_boost.eta.to_numpy()[:,np.newaxis], 
            e_jet1_boost.eta.to_numpy()[:,np.newaxis], 
            e_jet2_boost.eta.to_numpy()[:,np.newaxis], 
            e_jet3_boost.eta.to_numpy()[:,np.newaxis], 
            e_jet4_boost.eta.to_numpy()[:,np.newaxis], 
            e_jet5_boost.eta.to_numpy()[:,np.newaxis],
            e_jet0_boost.phi.to_numpy()[:,np.newaxis], 
            e_jet1_boost.phi.to_numpy()[:,np.newaxis], 
            e_jet2_boost.phi.to_numpy()[:,np.newaxis], 
            e_jet3_boost.phi.to_numpy()[:,np.newaxis], 
            e_jet4_boost.phi.to_numpy()[:,np.newaxis], 
            e_jet5_boost.m.to_numpy()[:,np.newaxis],
            e_jet0_boost.m.to_numpy()[:,np.newaxis], 
            e_jet1_boost.m.to_numpy()[:,np.newaxis], 
            e_jet2_boost.m.to_numpy()[:,np.newaxis], 
            e_jet3_boost.m.to_numpy()[:,np.newaxis], 
            e_jet4_boost.m.to_numpy()[:,np.newaxis], 
            e_jet5_boost.m.to_numpy()[:,np.newaxis]))

        n_signal = len(xtra_signal_features)
        n_excess = len(xtra_excess_features)

        signal_targets = np.tile([1,0], (n_signal, 1))
        excess_targets = np.tile([0,1], (n_excess, 1))

        # n_signal = ak.count(tree.jet_idx[mixed_ind][excess_signal_mask], axis=1)
        # excess_targets = np.zeros((nevents, 7), dtype=int)
        # excess_targets[np.arange(nevents), n_signal] += 1

        # signal_targets = np.zeros_like(excess_targets)
        # signal_targets[:,-1] += 1

        assert (np.any(~signal_targets[:,1]))
        assert (np.all(signal_targets[:,0]))
        assert (np.all(excess_targets[:,1]))
        assert (~np.any(excess_targets[:,0]))

        # Concatenate pT, eta, phi, and btag score
        signal_features = np.concatenate((
            signal_p4.pt.to_numpy(), 
            signal_p4.pt.to_numpy()**2, 
            signal_p4.eta.to_numpy(), 
            signal_p4.eta.to_numpy()**2, 
            signal_p4.phi.to_numpy(), 
            signal_p4.phi.to_numpy()**2, 
            signal_btag, 
            xtra_signal_features), axis=1)
        excess_features = np.concatenate((
            excess_p4.pt.to_numpy(), 
            excess_p4.pt.to_numpy()**2, 
            excess_p4.eta.to_numpy(), 
            excess_p4.eta.to_numpy()**2, 
            excess_p4.phi.to_numpy(), 
            excess_p4.phi.to_numpy()**2, 
            excess_btag, 
            xtra_excess_features), axis=1)

        assert (signal_features.shape == excess_features.shape)

        nan_mask = np.isnan(signal_features).any(axis=1)

        features = np.row_stack((signal_features, excess_features))
        good_mask = ~np.isnan(features).any(axis=1)
        self.features = features[good_mask, :]
        targets = np.row_stack((signal_targets, excess_targets))
        self.targets = targets[good_mask, :]

        assert len(self.features) == len(self.targets)
        assert np.isnan(self.features).sum() == 0

    def get_pair_p4(self, mask=None):
        # if signal: mask = self.jet_idx > -1
        # else: mask = self.jet_idx == -1
        jet1_pt, jet2_pt = ak.unzip(ak.combinations(self.tree.jet_pt[mask], 2))
        jet1_eta, jet2_eta = ak.unzip(ak.combinations(self.tree.jet_eta[mask], 2))
        jet1_phi, jet2_phi = ak.unzip(ak.combinations(self.tree.jet_phi[mask], 2))
        jet1_m, jet2_m = ak.unzip(ak.combinations(self.tree.jet_m[mask], 2))

        jet1_p4 = vector.obj(
            pt  = jet1_pt,
            eta = jet1_eta,
            phi = jet1_phi,
            m   = jet1_m)

        jet2_p4 = vector.obj(
            pt  = jet2_pt,
            eta = jet2_eta,
            phi = jet2_phi,
            m   = jet2_m)

        return jet1_p4, jet2_p4

class TrainTwo():

    def __init__(self, filename):

        tree = Tree(filename, 'sixBtree', as_ak=True)
        nevents = len(tree.jet_pt)

        n_sixb = tree.n_sixb
        local_ind = ak.local_index(tree.jet_idx)
        signal_jet_mask = tree.jet_idx > -1
        signal_jet_ind  = local_ind[signal_jet_mask]
        excess_jet_ind  = local_ind[~signal_jet_mask]

        HX_mask, H1_mask, H2_mask = self.Higgs_masks(tree.jet_idx)
        H_mask = HX_mask & H1_mask & H2_mask

        mixed_ind = ak.sort(ak.concatenate((excess_jet_ind, signal_jet_ind), axis=1)[:, :6], axis=1)
        mixed_ind_np = mixed_ind.to_numpy()

        signal_p4 = vector.obj(
            pt  = tree.jet_pt[signal_jet_mask],
            eta = tree.jet_eta[signal_jet_mask],
            phi = tree.jet_phi[signal_jet_mask],
            m   = tree.jet_m[signal_jet_mask]
        )

        excess_p4 = vector.obj(
            pt  = tree.jet_pt[mixed_ind],
            eta = tree.jet_eta[mixed_ind],
            phi = tree.jet_phi[mixed_ind],
            m   = tree.jet_m[mixed_ind]
        )

        signal_btag = tree.jet_btag[signal_jet_mask].to_numpy()
        excess_btag = tree.jet_btag[mixed_ind].to_numpy()

        s_jet0_boosted_pt, s_jet1_boosted_pt, s_jet2_boosted_pt, s_jet3_boosted_pt, s_jet4_boosted_pt, s_jet5_boosted_pt = self.get_boosted(tree, signal_jet_mask)
        e_jet0_boosted_pt, e_jet1_boosted_pt, e_jet2_boosted_pt, e_jet3_boosted_pt, e_jet4_boosted_pt, e_jet5_boosted_pt = self.get_boosted(tree, mixed_ind)

        # Concatenate pT, eta, phi, and btag score
        signal_inputs = np.concatenate((signal_p4.pt.to_numpy(), signal_p4.eta.to_numpy(), signal_p4.phi.to_numpy(), signal_btag, s_jet0_boosted_pt, s_jet1_boosted_pt, s_jet2_boosted_pt, s_jet3_boosted_pt, s_jet4_boosted_pt, s_jet5_boosted_pt), axis=1)
        excess_inputs = np.concatenate((excess_p4.pt.to_numpy(), excess_p4.eta.to_numpy(), excess_p4.phi.to_numpy(), excess_btag, e_jet0_boosted_pt, e_jet1_boosted_pt, e_jet2_boosted_pt, e_jet3_boosted_pt, e_jet4_boosted_pt, e_jet5_boosted_pt), axis=1)
        print(signal_inputs.shape)

        self.signal_features = signal_inputs
        self.excess_features = excess_inputs

        def Higgs_masks(self, jet_idx):

            mask0 = ak.where(jet_idx == 0, 1, 0) # find HX b1
            mask1 = ak.where(jet_idx == 1, 1, 0) # find HX b1
            HX_mask = ak.where(mask1, 1, mask0) # include HX b2

            mask2 = ak.where(jet_idx == 2, 1, 0) # find HY1 b1
            mask3 = ak.where(jet_idx == 3, 1, 0) # find HY1 b1
            H1_mask = ak.where(mask3, 1, mask2) # include HY1 b2

            mask4 = ak.where(jet_idx == 4, 1, 0)# find HY2 b1
            mask5 = ak.where(jet_idx == 5, 1, 0)# find HY2 b1
            H2_mask = ak.where(mask5, 1, mask4)# include HY2 b2

        return  HX_mask, H1_mask, H2_mask

class combos():

    def __init__(self, filename, njets=7, tag6b=None, tag2b=None):

        self.evaluate_6b(filename, njets, tag6b)
        # self.evaluate_2b(filename, njets, tag2b)``

    def evaluate_2b(self, filename, njets, tag):
        print("Loading...")
        tree = Tree(filename, 'sixBtree', as_ak=True)
        nevents = len(tree.jet_pt)

        # Arrays of indices representing the pt-ordered event
        jet_ptordered = ak.argsort(tree.jet_pt, ascending=False)
        # Arrays containing indices representing all 6-jet combinations in each event
        jet_comb = ak.combinations(jet_ptordered, 2)
        # Unzip the combinations into their constituent jets
        jet0, jet1 = ak.unzip(jet_comb)
        # Zip the constituents back together
        combos = ak.concatenate([jeti[:,:,np.newaxis] for jeti in (jet0, jet1)], axis=-1)

        print("Broadcasting...")
        pt1, pt2 = ak.unzip(ak.combinations(tree.jet_pt, njets))
        eta1, eta2 = ak.unzip(ak.combinations(tree.jet_eta, njets))
        phi1, phi2 = ak.unzip(ak.combinations(tree.jet_phi, njets))
        m1, m2 = ak.unzip(ak.combinations(tree.jet_m, njets))
        btag1, btag2 = ak.unzip(ak.combinations(tree.jet_btag, njets))
        idx1, idx2 = ak.unzip(ak.combinations(tree.jet_idx, njets))

        # jets = [jet0, jet1]
        # [jet0_boosted_pt, jet1_boosted_pt] = self.get_boosted_p4(tree, jets)
        jet0_p4 = p4(tree, jet0)
        jet1_p4 = p4(tree, jet1)
        jet_ids = combos.to_numpy() # (nevents, ncombos, 2) array of jet ids in a pair
        deltaR = ak.flatten(jet0_p4.deltaR(jet1_p4)).to_numpy()

        print("Concatenating inputs...")
        # Concatenate pT, eta, phi, and btag score
        inputs = np.concatenate((pt, eta, phi, btag), axis=2)
        D1 = inputs.shape[0]
        D2 = inputs.shape[1]
        D3 = inputs.shape[2]

        print("Defining features...")
        # Reshape inputs to match the structure of the boosted pTs and concatenate all of it together
        features = np.column_stack((inputs.reshape(D1*D2, D3), deltaR))

        print("Labeling signal...")
        # jet_idx provides the matching index (-1 if unmatched) to gen jets
        idx_shape = idx[combos].shape
        # Signal mask for combinations
        combo_idx = idx.reshape(idx_shape[0]*idx_shape[1], idx_shape[2])
        HX_mask = np.all(np.logical_or(combo_idx == 0, combo_idx == 1), axis=1)
        H1_mask = np.all(np.logical_or(combo_idx == 2, combo_idx == 3), axis=1)
        H2_mask = np.all(np.logical_or(combo_idx == 4, combo_idx == 5), axis=1)
        Higgs_mask = np.logical_or(np.logical_or(HX_mask, H1_mask), H2_mask)
        evt_mask = Higgs_mask.reshape(nevents, ncombos)
        sgnl_ind = np.argwhere(Higgs_mask) # (length = nevents where sgnl is found)

        print("Loading model...")
        # Load scaler and model
        scaler, model = load_model(location='../../2jet_classifier/', tag=tag)
        # Generate predictions by applying the model to the features
        scores = model.predict(scaler.transform(features))[:,0]

        # Change into shape of events containing 7 distinct combinations
        scores_reshaped = scores.reshape(nevents, ncombos)

        # Return maximum score found in the event
        max_evt_mask = np.argsort(scores_reshaped, axis=1)[:,::-1,np.newaxis]
        jet_ids_by_score = np.take_along_axis(jet_ids, max_evt_mask, 1)
        max_evt_score = np.sort(scores_reshaped, axis=1)[:,::-1]

        self.obtain_unique_pairs(jet_ids_by_score)

        sgnl_is_max = np.equal(sgnl_ind, max_evt_mask[evt_mask])
        print(sgnl_is_max.sum()/len(sgnl_is_max))

    def obtain_unique_pairs(self, sorted_jet_ids):
        distinct_pairs = np.ones((sorted_jet_ids.shape[0], 3)) * -1
        print(distinct_pairs)
        sys.exit()
        for _ in range(sorted_jet_ids.shape[1]):
            sorted_jet_ids

    def evaluate_6b(self, filename, njets, tag):

        print("Initializing...")

        tree = Tree(filename, 'sixBtree', training=True)
        nevents = len(tree.jet_pt)

        print("Indexing...")

        jet1pt, jet2pt, jet3pt, jet4pt, jet5pt, jet6pt = ak.unzip(ak.combinations(tree.jet_pt, njets))
        sys.exit()




        # Arrays of indices representing the pt-ordered event
        # jet_ptordered = ak.argsort(tree.jet_pt, ascending=False)
        # Arrays containing indices representing all 6-jet combinations in each event
        jet_index = ak.local_index(tree.jet_pt)
        jet_comb = ak.combinations(jet_index, njets)
        # Unzip the combinations into their constituent jets
        jet0, jet1, jet2, jet3, jet4, jet5 = ak.unzip(jet_comb)
        # Zip the constituents back together
        combos = ak.concatenate([jeti[:,:,np.newaxis] for jeti in (jet0, jet1, jet2, jet3, jet4, jet5)], axis=-1)

        pt1, pt2 = ak.unzip(ak.combinations(tree.jet_pt, njets))
        eta1, eta2 = ak.unzip(ak.combinations(tree.jet_eta, njets))
        phi1, phi2 = ak.unzip(ak.combinations(tree.jet_phi, njets))
        m1, m2 = ak.unzip(ak.combinations(tree.jet_m, njets))
        btag1, btag2 = ak.unzip(ak.combinations(tree.jet_btag, njets))
        idx1, idx2 = ak.unzip(ak.combinations(tree.jet_idx, njets))

        signal_mask = idx > -1
        n_signal = ak.count(idx[signal_mask], axis=2)
        n_signal = n_signal.to_numpy()
        self.n_signal = n_signal.reshape(n_signal.shape[0]*n_signal.shape[1])
        idx = idx.to_numpy()

        jets = [jet0, jet1, jet2, jet3, jet4, jet5]
        [jet0_boosted_pt, jet1_boosted_pt, jet2_boosted_pt, jet3_boosted_pt, jet4_boosted_pt, jet5_boosted_pt] = get_boosted_p4(tree, jets)

        # Concatenate pT, eta, phi, and btag score
        inputs = np.concatenate((pt, eta, phi, btag), axis=2)
        D1 = inputs.shape[0]
        D2 = inputs.shape[1]
        D3 = inputs.shape[2]

        # Reshape inputs to match the structure of the boosted pTs and concatenate all of it together
        features = np.concatenate((inputs.reshape(D1*D2, D3), jet0_boosted_pt, jet1_boosted_pt, jet2_boosted_pt, jet3_boosted_pt, jet4_boosted_pt, jet5_boosted_pt), axis=1)
        
        # jet_idx provides the matching index (-1 if unmatched) to gen jets
        idx_shape = ak.to_numpy(idx).shape
        # Signal mask for combinations
        signal_mask = np.all(ak.to_numpy(idx).reshape(idx_shape[0]*idx_shape[1], idx_shape[2]) > -1, axis=1)
        evt_mask = signal_mask.reshape(nevents, njets)
        evt_sgnl_mask = np.any(evt_mask, axis=1)
        sgnl_ind = np.argwhere(evt_mask)[:,1] # (length = nevents where sgnl is found)

        # Load scaler and model
        scaler, model = load_model(location='../', tag=tag)
        # Generate predictions by applying the model to the features
        scores = model.predict(scaler.transform(features))
        self.scores = scores
        self.category = np.argmax(scores, axis=1)

        # Change into shape of events containing 7 distinct combinations
        # scores_reshaped = scores.reshape(nevents, njets)

        # Return maximum score found in the event
        # max_evt_mask = scores_reshaped.argmax(axis=1)
        # max_evt_score = scores_reshaped.max(axis=1)

        # sgnl_is_max = np.equal(sgnl_ind, max_evt_mask[evt_sgnl_mask])
        # print(sgnl_is_max.sum()/len(sgnl_is_max))

        

    def apply_6j_model(self, features, tag, location='../'):

        print("Applying 6b Model to combinations. Please wait.")
        scaler, model = load_model(location, tag)
        test_features = scaler.transform(combo_features)
        scores_combo = model.predict(test_features)[:,0]

        print("Selecting highest scoring combination from each event.")
        # self.select_highest_scoring_combos()


