from . import *

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