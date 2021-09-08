from . import *
from lbn import LBN

class TrainLBN():

    def get_lbn_features(self, ncombos, mask):
        jet_pt = training.get_t6('jet_pt')
        jet_eta = training.get_t6('jet_eta')
        jet_phi = training.get_t6('jet_phi')
        jet_m = training.get_t6('jet_m')
        jet_btag = training.get_t6('jet_btag')

        pt_list  = ak.unzip(ak.combinations(jet_pt[mask], ncombos))
        eta_list = ak.unzip(ak.combinations(jet_eta[mask], ncombos))
        phi_list = ak.unzip(ak.combinations(jet_phi[mask], ncombos))
        m_list   = ak.unzip(ak.combinations(jet_m[mask], ncombos))
        btag = jet_btag[mask].to_numpy()

        pt_list = [ak.flatten(pt) for pt in pt_list]
        eta_list = [ak.flatten(eta) for pt in eta_list]
        phi_list = [ak.flatten(phi) for pt in phi_list]
        m_list = [ak.flatten(m) for pt in m_list]

        p4_list = [self.p4(pt,eta,phi,m) for pt,eta,phi,m in zip(pt_list,eta_list,phi_list,m_list)]
        input = [[p4.E, p4.px, p4.py, p4.pz] for p4 in p4_list]
        input = np.asarray(input)

        lbn_features = lbn(np.transpose(input, (2,0,1)), features=["E","px","py","pz"])
        lbn_features = tf.concat([lbn_features, btag], axis=1)

    def __init__(self, filename, n_particles=11, n_restframes=5, boost_mode=LBN.PRODUCT):
        lbn = LBN(n_particles, n_restframes, boost_mode=boost_mode)
        training = TrainSix(filename)
        signal = training.correct_mask
        cbkgd  = training.incorrect_mask
        
        signal_lbn_features = self.get_lbn_features(6, signal)
        bkgd_lbn_features = self.get_lbn_features(6, cbkgd)
    
        features = tf.concat([signal_lbn_features, bkgd_lbn_features], axis=0)
        targets = np.concatenate((
            np.tile([1,0], (signal_lbn_features.get_shape()[0], 1)),
            np.tile([0,1], (bkgd_lbn_features.get_shape()[0], 1))
        ))

    def p4(self, pt, eta, phi, m):
        return vector.obj(pt=pt, eta=eta, phi=phi, m=m)

class TrainSix():

    def __init__(self, filename):

        print("Initializing tree.")
        tree = Tree(filename, 'sixBtree', training=True)
        self.tree = tree
        nevents = len(tree.jet_pt)

        print("Identifying signal.")
        t6_jet_idx = tree.jet_idx[:, :6]
        t6_n_signal = ak.sum(t6_jet_idx > -1, axis=1)
        t6_signal = ak.all(t6_jet_idx > -1, axis=1)
        t6_incorrect = ~t6_signal

        n,e = np.histogram(t6_n_signal.to_numpy(), bins=range(8), density=1)
        t6_n_wrong = (t6_signal.to_numpy().sum()*n[:-1]/n[:-1].sum()).astype(int)

        incorrect_mask = []
        for i in range(6):
            n_mask = t6_n_signal == i
            events = np.arange(nevents)[n_mask]
            events = np.random.choice(events, t6_n_wrong[i], replace=False)
            incorrect_mask.extend(events)
        rd.shuffle(incorrect_mask)

        self.incorrect_mask = ak.Array(incorrect_mask)
        self.correct_mask = ak.local_index(t6_signal)[t6_signal]

        jet_pt = self.get_t6('jet_pt')
        jet_eta = self.get_t6('jet_eta')
        jet_phi = self.get_t6('jet_phi')
        jet_m = self.get_t6('jet_m')
        jet_btag = self.get_t6('jet_btag')

        # signal_kinematics = 

    def get_t6(self, branch):
        return getattr(self.tree, branch)[:, :6]

    def get_min_dR(self, mask=None):
        jet1, jet2 = self.get_pair_p4(mask)
        all_dR = jet1.deltaR(jet2)
        min_dR = ak.min(all_dR, axis=1)
        print(min_dR)
        return min_dR.to_numpy()

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