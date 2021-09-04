"""
Project: 6b Final States - 
Author: Suzanne Rosenzweig

This class sorts MC-generated events into an array of input features for use in training a neural network and in evaluating the performance of the model using a test set of examples.

Notes:
Training samples are prepared such that these requirements are already imposed:
- n_jet > 6
- n_sixb == 6
"""

from . import *

# Standard library imports
from math import comb
import sys  # JAKE DELETE
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

def broadcast(jet_arr, njets=7):
    # In order to use the combos array as a mask, the jet arrays must be broadcast into the same shape, or vice versa
    # Awkward arrays don't broadcast the same way NumPy arrays do
    # See this discussion: https://github.com/scikit-hep/awkward-0.x/issues/253
    # Jim Pivarski recommends converting to NumPy arrays to broadcast
    return ak.from_numpy(np.repeat(jet_arr[:,np.newaxis].to_numpy(), njets, axis=1))

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
    def __init__(self, filename, treename='sixBtree'):
        """
        A class for handling TTrees. 
        """

        tree = uproot.open(f"{filename}:{treename}")
        self.tree = tree
        for k, v in tree.items():
            setattr(self, k, v.array())

        self.local_ind = ak.local_index(self.jet_pt)

        self.signal_evt_mask = ak.sum(self.jet_signalId > -1, axis=1) == 6
        self.signal_jet_mask = self.jet_signalId[self.signal_evt_mask] > -1

        for k, v in tree.items():
            try: setattr(self, 'signal_' + k, v.array()[self.signal_evt_mask][self.signal_jet_mask])
            except: continue

    def sort(self, mask):
        for k, v in self.tree.items():
            try: setattr(self, k, v.array()[mask])
            except: continue

    def keys(self):
        print(self.tree.keys())

    def get(self, key):
        return tree[key]

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

        jet0_boosted_pt = jet0_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        jet1_boosted_pt = jet1_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        jet2_boosted_pt = jet2_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        jet3_boosted_pt = jet3_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        jet4_boosted_pt = jet4_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        jet5_boosted_pt = jet5_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]

        return jet0_boosted_pt, jet1_boosted_pt, jet2_boosted_pt, jet3_boosted_pt, jet4_boosted_pt, jet5_boosted_pt, jet6_p4

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

        tree = Tree(filename, 'sixBtree', as_ak=True)
        if dijet:
            sort_mask = ak.argsort(tree.jet_idx, axis=1)
            tree.sort(sort_mask)
        nevents = len(tree.jet_pt)

        n_sixb = tree.n_sixb
        local_ind = ak.local_index(tree.jet_idx)
        signal_jet_mask = tree.jet_idx > -1
        signal_jet_ind  = local_ind[signal_jet_mask]
        excess_jet_ind  = local_ind[~signal_jet_mask]

        if dijet:
            HX_b1_mask = tree.jet_idx == 0
            HX_b2_mask = tree.jet_idx == 1
            HX_mask = HX_b1_mask | HX_b2_mask
            H1_b1_mask = tree.jet_idx == 2
            H1_b2_mask = tree.jet_idx == 3
            H1_mask = H1_b1_mask | H1_b2_mask
            H2_b1_mask = tree.jet_idx == 4
            H2_b2_mask = tree.jet_idx == 5
            H2_mask = H2_b1_mask | H2_b2_mask

            sorted_HX_mask = ak.argsort(tree.jet_pt[HX_mask])[:,::-1]
            sorted_H1_mask = ak.argsort(tree.jet_pt[H1_mask])[:,::-1] + 2
            sorted_H2_mask = ak.argsort(tree.jet_pt[H2_mask])[:,::-1] + 4

            sorted_HX_ind = signal_jet_ind[sorted_HX_mask]
            sorted_H1_ind = signal_jet_ind[sorted_H1_mask]
            sorted_H2_ind = signal_jet_ind[sorted_H2_mask]

            signal_jet_ind = ak.concatenate((sorted_HX_ind, sorted_H1_ind, sorted_H2_ind), axis=1)

            # A note about the following few lines of code
            # In order to randomize the rows, I had to convert to NumPy objects, shuffle, then convert back to Awkward Arrays
            # But since NumPy arrays must be regularly shaped (no jaggedness), the result is a RegularArray
            # APPARENTLY the behavior of tree.branch[RegularArray] is different than tree.branch[IrregularArray]
            # And this caused me MUCH strife. I figured it out though. Good. God.
            random_ak_ind = np.asarray([np.random.choice(each_row, 6, replace=False) for each_row in signal_jet_ind.to_numpy()])
            random_ak_ind = ak.from_regular(ak.from_numpy(random_ak_ind, regulararray=True))
            mixed_ind = tree.jet_idx[random_ak_ind]
            mixed_ind_np = mixed_ind.to_numpy()
        else:
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

        
        if dijet:
            s_jet0_p4, s_jet1_p4, s_jet2_p4, s_jet3_p4, s_jet4_p4, s_jet5_p4 = self.get_p4s(tree, signal_jet_mask)
            e_jet0_p4, e_jet1_p4, e_jet2_p4, e_jet3_p4, e_jet4_p4, e_jet5_p4 = self.get_p4s(tree, mixed_ind)

            s_dijet1 = s_jet0_p4 + s_jet1_p4
            s_dijet2 = s_jet2_p4 + s_jet3_p4
            s_dijet3 = s_jet4_p4 + s_jet5_p4

            e_dijet1 = e_jet0_p4 + e_jet1_p4
            e_dijet2 = e_jet2_p4 + e_jet3_p4
            e_dijet3 = e_jet4_p4 + e_jet5_p4

            H_pt_order = np.argsort(np.column_stack((
                s_dijet1.pt.to_numpy(), 
                s_dijet2.pt.to_numpy(), 
                s_dijet3.pt.to_numpy())), axis=1)[:,::-1]

            s_dijet1_dR = s_jet0_p4.deltaR(s_jet1_p4).to_numpy()
            s_dijet2_dR = s_jet2_p4.deltaR(s_jet3_p4).to_numpy()
            s_dijet3_dR = s_jet4_p4.deltaR(s_jet5_p4).to_numpy()
            s_dR_features = ak.from_regular(ak.from_numpy(np.column_stack((s_dijet1_dR, s_dijet2_dR, s_dijet3_dR))))
            s_dR_features = s_dR_features[ak.from_regular(ak.from_numpy(H_pt_order))].to_numpy()

            H_pt_order = np.argsort(np.column_stack((
                s_dijet1.pt.to_numpy(), 
                s_dijet2.pt.to_numpy(), 
                s_dijet3.pt.to_numpy())), axis=1)[:,::-1]

            excess_pt_order = np.argsort(np.column_stack((
                e_dijet1.pt.to_numpy(), 
                e_dijet2.pt.to_numpy(), 
                e_dijet3.pt.to_numpy())), axis=1)[:,::-1]

            fig, ax = plt.subplots(nrows=1, ncols=2)
            s_dijet1.mass, s_dijet2.mass, s_dijet3.mass
            e_dijet1.mass, e_dijet2.mass, e_dijet3.mass
            
            e_dijet1_dR = e_jet0_p4.deltaR(e_jet1_p4).to_numpy()
            e_dijet2_dR = e_jet2_p4.deltaR(e_jet3_p4).to_numpy()
            e_dijet3_dR = e_jet4_p4.deltaR(e_jet5_p4).to_numpy()
            e_dR_features = ak.from_regular(ak.from_numpy(np.column_stack((e_dijet1_dR, e_dijet2_dR, e_dijet3_dR))))
            e_dR_features = e_dR_features[ak.from_regular(ak.from_numpy(excess_pt_order))].to_numpy()

            H_pt_order = np.where(H_pt_order == 2, 4, H_pt_order)
            H_pt_order = np.where(H_pt_order == 1, 2, H_pt_order)
            excess_pt_order = np.where(excess_pt_order == 2, 4, excess_pt_order)
            excess_pt_order = np.where(excess_pt_order == 1, 2, excess_pt_order)

            H_pt_order = np.column_stack((H_pt_order[:,0], H_pt_order[:,0] + 1, H_pt_order[:,1], H_pt_order[:,1] + 1, H_pt_order[:,2], H_pt_order[:,2] + 1))
            H_pt_order = ak.from_regular(ak.from_numpy(H_pt_order))
            excess_pt_order = np.column_stack((excess_pt_order[:,0], excess_pt_order[:,0] + 1, excess_pt_order[:,1], excess_pt_order[:,1] + 1, excess_pt_order[:,2], excess_pt_order[:,2] + 1))
            excess_pt_order = ak.from_regular(ak.from_numpy(excess_pt_order))

            signal_jet_ind = signal_jet_ind[H_pt_order]
            mixed_ind = mixed_ind[excess_pt_order]

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

            xtra_signal_features = np.sort(np.column_stack((
                s_dijet1.pt.to_numpy(), 
                s_dijet2.pt.to_numpy(), 
                s_dijet3.pt.to_numpy())), axis=1)[:,::-1]

            xtra_signal_features = np.column_stack((xtra_signal_features, s_dR_features))

            xtra_excess_features = np.sort(np.column_stack((
                e_dijet1.pt.to_numpy(), 
                e_dijet2.pt.to_numpy(), 
                e_dijet3.pt.to_numpy())), axis=1)[:,::-1]

            xtra_excess_features = np.column_stack((xtra_excess_features, e_dR_features))
            
            signal_targets = np.tile([1,0], (nevents, 1))
            excess_targets = np.tile([0,1], (nevents, 1))

        else:
            s_jet0_boosted_pt, s_jet1_boosted_pt, s_jet2_boosted_pt, s_jet3_boosted_pt, s_jet4_boosted_pt, s_jet5_boosted_pt, s_jet6_p4 = self.get_boosted(tree, signal_jet_mask)
            e_jet0_boosted_pt, e_jet1_boosted_pt, e_jet2_boosted_pt, e_jet3_boosted_pt, e_jet4_boosted_pt, e_jet5_boosted_pt, e_jet6_p4 = self.get_boosted(tree, mixed_ind)

            xtra_signal_features = np.column_stack((
                s_jet0_boosted_pt, 
                s_jet1_boosted_pt, 
                s_jet2_boosted_pt, 
                s_jet3_boosted_pt, 
                s_jet4_boosted_pt, 
                s_jet5_boosted_pt))

            xtra_excess_features = np.column_stack((
                e_jet0_boosted_pt, 
                e_jet1_boosted_pt, 
                e_jet2_boosted_pt, 
                e_jet3_boosted_pt, 
                e_jet4_boosted_pt, 
                e_jet5_boosted_pt))

            n_signal = ak.count(tree.jet_idx[mixed_ind][excess_signal_mask], axis=1)
            excess_targets = np.zeros((nevents, 7), dtype=int)
            excess_targets[np.arange(nevents), n_signal] += 1

            signal_targets = np.zeros_like(excess_targets)
            signal_targets[:,-1] += 1

            assert (np.any(~signal_targets[:,:-1]))
            assert (np.all(signal_targets[:,-1]))
        # assert (np.all(excess_targets[:,1]))
        # assert (~np.any(excess_targets[:,0]))

        # Concatenate pT, eta, phi, and btag score
        signal_features = np.concatenate((
            signal_p4.pt.to_numpy(), 
            signal_p4.eta.to_numpy(), 
            signal_p4.phi.to_numpy(), 
            signal_btag, 
            xtra_signal_features), axis=1)
        excess_features = np.concatenate((
            excess_p4.pt.to_numpy(), 
            excess_p4.eta.to_numpy(), 
            excess_p4.phi.to_numpy(), 
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


class dijet():

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

        jet0_boosted_pt = jet0_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        jet1_boosted_pt = jet1_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        jet2_boosted_pt = jet2_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        jet3_boosted_pt = jet3_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        jet4_boosted_pt = jet4_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]
        jet5_boosted_pt = jet5_p4.boost_p4(jet6_p4).pt.to_numpy()[:,np.newaxis]

        return jet0_boosted_pt, jet1_boosted_pt, jet2_boosted_pt, jet3_boosted_pt, jet4_boosted_pt, jet5_boosted_pt, jet6_p4

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

    #### BROKEN
    def __init__():

        tree = Tree(filename, 'sixBtree', as_ak=True)
        sort_mask = ak.argsort(tree.jet_idx, axis=1)
        tree.sort(sort_mask)
        nevents = len(tree.jet_pt)

        n_sixb = tree.n_sixb
        local_ind = ak.local_index(tree.jet_idx)
        signal_jet_mask = tree.jet_idx > -1
        signal_jet_ind  = local_ind[signal_jet_mask]

        HX_b1_mask = tree.jet_idx == 0
        HX_b2_mask = tree.jet_idx == 1
        HX_mask = HX_b1_mask | HX_b2_mask
        H1_b1_mask = tree.jet_idx == 2
        H1_b2_mask = tree.jet_idx == 3
        H1_mask = H1_b1_mask | H1_b2_mask
        H2_b1_mask = tree.jet_idx == 4
        H2_b2_mask = tree.jet_idx == 5
        H2_mask = H2_b1_mask | H2_b2_mask

        sorted_HX_mask = ak.argsort(tree.jet_pt[HX_mask])[:,::-1]
        sorted_H1_mask = ak.argsort(tree.jet_pt[H1_mask])[:,::-1] + 2
        sorted_H2_mask = ak.argsort(tree.jet_pt[H2_mask])[:,::-1] + 4

        sorted_HX_ind = signal_jet_ind[sorted_HX_mask]
        sorted_H1_ind = signal_jet_ind[sorted_H1_mask]
        sorted_H2_ind = signal_jet_ind[sorted_H2_mask]

        signal_jet_ind = ak.concatenate((sorted_HX_ind, sorted_H1_ind, sorted_H2_ind), axis=1)
        jet_comb = ak.combinations(jet_ptordered, 2)
        # Unzip the combinations into their constituent jets
        jet0, jet1, jet2, jet3, jet4, jet5 = ak.unzip(jet_comb)

        #### WORKING HERE
        # Zip the constituents back together
        combos = ak.concatenate([jeti[:,:,np.newaxis] for jeti in (jet0, jet1)], axis=-1)

        signal_p4 = vector.obj(
            pt  = tree.jet_pt[signal_jet_ind],
            eta = tree.jet_eta[signal_jet_ind],
            phi = tree.jet_phi[signal_jet_ind],
            m   = tree.jet_m[signal_jet_ind]
        )

        signal_btag = tree.jet_btag[signal_jet_ind].to_numpy()

        jet0_p4, jet1_p4, jet2_p4, jet3_p4, jet4_p4, jet5_p4 = self.get_p4s(tree, signal_jet_mask)

        dijet1 = jet0_p4 + jet1_p4
        dijet2 = jet2_p4 + jet3_p4
        dijet3 = jet4_p4 + jet5_p4

        H_pt_order = np.argsort(np.column_stack((
            dijet1.pt.to_numpy(), 
            dijet2.pt.to_numpy(), 
            dijet3.pt.to_numpy())), axis=1)[:,::-1]

        dijet1_dR = jet0_p4.deltaR(jet1_p4).to_numpy()
        dijet2_dR = jet2_p4.deltaR(jet3_p4).to_numpy()
        dijet3_dR = jet4_p4.deltaR(jet5_p4).to_numpy()
        dR_features = ak.from_regular(ak.from_numpy(np.column_stack((dijet1_dR, dijet2_dR, dijet3_dR))))
        dR_features = dR_features[ak.from_regular(ak.from_numpy(H_pt_order))].to_numpy()

        H_pt_order = np.argsort(np.column_stack((
            dijet1.pt.to_numpy(), 
            dijet2.pt.to_numpy(), 
            dijet3.pt.to_numpy())), axis=1)[:,::-1]

        fig, ax = plt.subplots(nrows=1, ncols=2)
        dijet1.mass, dijet2.mass, dijet3.mass

        H_pt_order = np.where(H_pt_order == 2, 4, H_pt_order)
        H_pt_order = np.where(H_pt_order == 1, 2, H_pt_order)

        H_pt_order = np.column_stack((H_pt_order[:,0], H_pt_order[:,0] + 1, H_pt_order[:,1], H_pt_order[:,1] + 1, H_pt_order[:,2], H_pt_order[:,2] + 1))
        H_pt_order = ak.from_regular(ak.from_numpy(H_pt_order))

        signal_jet_ind = signal_jet_ind[H_pt_order]

        signal_p4 = vector.obj(
            pt  = tree.jet_pt[signal_jet_ind],
            eta = tree.jet_eta[signal_jet_ind],
            phi = tree.jet_phi[signal_jet_ind],
            m   = tree.jet_m[signal_jet_ind]
        )

        signal_btag = tree.jet_btag[signal_jet_ind].to_numpy()
        excesbtag = tree.jet_btag[mixed_ind].to_numpy()

        xtra_signal_features = np.sort(np.column_stack((
            dijet1.pt.to_numpy(), 
            dijet2.pt.to_numpy(), 
            dijet3.pt.to_numpy())), axis=1)[:,::-1]

        xtra_signal_features = np.column_stack((xtra_signal_features, dR_features))

        # Concatenate pT, eta, phi, and btag score
        signal_features = np.concatenate((
            signal_p4.pt.to_numpy(), 
            signal_p4.eta.to_numpy(), 
            signal_p4.phi.to_numpy(), 
            signal_btag, 
            xtra_signal_features), axis=1)

        assert (signal_features.shape == excess_features.shape)

        nan_mask = np.isnan(signal_features).any(axis=1)

        features = np.row_stack((signal_features, excess_features))
        good_mask = ~np.isnan(features).any(axis=1)
        self.features = features[good_mask, :]
        targets = np.row_stack((signal_targets, excess_targets))
        self.targets = targets[good_mask, :]

        assert len(self.features) == len(self.targets)
        assert np.isnan(self.features).sum() == 0

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
        # self.evaluate_2b(filename, njets, tag2b)

    def evaluate_2b(self, filename, njets, tag):
        print("Loading...")
        tree = Tree(filename, 'sixBtree', as_ak=True)
        nevents = len(tree.jet_pt)
        ncombos = comb(njets, 2)

        # Arrays of indices representing the pt-ordered event
        jet_ptordered = ak.argsort(tree.jet_pt, ascending=False)
        # Arrays containing indices representing all 6-jet combinations in each event
        jet_comb = ak.combinations(jet_ptordered, 2)
        # Unzip the combinations into their constituent jets
        jet0, jet1 = ak.unzip(jet_comb)
        # Zip the constituents back together
        combos = ak.concatenate([jeti[:,:,np.newaxis] for jeti in (jet0, jet1)], axis=-1)

        print("Broadcasting...")
        pt = broadcast(tree.jet_pt, ncombos)[combos].to_numpy()
        eta = broadcast(tree.jet_eta, ncombos)[combos].to_numpy()
        phi = broadcast(tree.jet_phi, ncombos)[combos].to_numpy()
        m = broadcast(tree.jet_m, ncombos)[combos].to_numpy()
        btag = broadcast(tree.jet_btag, ncombos)[combos].to_numpy()
        idx = broadcast(tree.jet_idx, ncombos)[combos].to_numpy()

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

        tree = Tree(filename, 'sixBtree', as_ak=True)
        nevents = len(tree.jet_pt)

        # Arrays of indices representing the pt-ordered event
        # jet_ptordered = ak.argsort(tree.jet_pt, ascending=False)
        # Arrays containing indices representing all 6-jet combinations in each event
        jet_index = ak.local_index(tree.jet_pt)
        jet_comb = ak.combinations(jet_index, 6)
        # Unzip the combinations into their constituent jets
        jet0, jet1, jet2, jet3, jet4, jet5 = ak.unzip(jet_comb)
        # Zip the constituents back together
        combos = ak.concatenate([jeti[:,:,np.newaxis] for jeti in (jet0, jet1, jet2, jet3, jet4, jet5)], axis=-1)

        pt = broadcast(tree.jet_pt, njets)[combos].to_numpy()
        eta = broadcast(tree.jet_eta, njets)[combos].to_numpy()
        phi = broadcast(tree.jet_phi, njets)[combos].to_numpy()
        m = broadcast(tree.jet_m, njets)[combos].to_numpy()
        btag = broadcast(tree.jet_btag, njets)[combos].to_numpy()
        idx = broadcast(tree.jet_idx, njets)[combos]
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


    


class EventShapes():

    def __init__(self, filename):
        
        tree = Tree(filename)
        