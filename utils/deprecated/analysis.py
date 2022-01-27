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
