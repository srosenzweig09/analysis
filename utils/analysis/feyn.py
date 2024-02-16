import json, uproot
import numpy as np
import awkward as ak
from utils.analysis.particle import Particle, Higgs, Y

old_model_name = '20230731_7d266883bbfb88fe4e226783a7d1c9db_ranger_lr0.0047_batch2000_withbkg'
old_model_path = f'/eos/uscms/store/user/srosenzw/weaver/models/exp_tree_official/feynnet_ranker_6b/{old_model_name}/predict_output'

new_model_name = 'version_23183119'
# new_model_path = f"/eos/uscms/store/user/srosenzw/weaver/cmsuf/data/store/user/srosenzw/lightning/models/feynnet_lightning/X_YH_3H_6b/x3h/lightning_logs/{new_model_name}/predict/"
new_model_path = f"/cmsuf/data/store/user/srosenzw/lightning/models/feynnet_lightning/X_YH_3H_6b/x3h/lightning_logs/{new_model_name}/predict/"
new_model_path = f"/cmsuf/data/store/user/srosenzw/lightning/models/feynnet_lightning/X_YH_3H_6b/x3h/lightning_logs/{new_model_name}/predict/"

def getMassDict(year):
    with open(f"{new_model_path}/{year}/samples.json", 'r') as file:
        mass_dict = json.load(file)
    return mass_dict

class Model():

    def __init__(self, which, tree):
        if which == 'new': self.init_new_model(tree)
        elif which == 'old': self.init_old_model(tree)
        else: raise ValueError(f"Model type '{which}' not recognized")

        self.init_particles(tree)
        self.copy_attributes(tree)

    def init_new_model(self, tree):
        print(".. initializing new FeynNet model")
        mass_dict = getMassDict(tree.year_long)
        model_path = mass_dict[tree.filepath]
        # model_path = f"{new_model_path}/{tree.year}/{hash}.root"
        with uproot.open(model_path) as f:
            t = f['Events']
            # print(t['sorted_rank'].array())
            maxcomb = ak.firsts(t['sorted_j_assignments'].array())
        self.combos = ak.from_regular(maxcomb)
        self.model_name = new_model_name
        # print(print(combos))

    def init_old_model(self, tree):
        print(".. initializing old FeynNet model")
        model_path = f"{old_model_path}/{tree.year}/{tree.filename}.root"
        with uproot.open(model_path) as f:
            f = f['Events']
            maxcomb = f['max_comb'].array(library='np')

        combos = maxcomb.astype(int)
        self.combos = ak.from_regular(combos)
        self.model_name = old_model_name

    def init_particles(self, tree):
        btag_mask = ak.argsort(tree.jet_btag, axis=1, ascending=False) < 6

        pt = tree.jet_ptRegressed[btag_mask][self.combos]
        phi = tree.jet_phi[btag_mask][self.combos]
        eta = tree.jet_eta[btag_mask][self.combos]
        m = tree.jet_mRegressed[btag_mask][self.combos]
        btag = tree.jet_btag[btag_mask][self.combos]
        sig_id = tree.jet_signalId[btag_mask][self.combos]
        h_id = (tree.jet_signalId[btag_mask][self.combos] + 2) // 2

        self.btag_avg = ak.mean(btag, axis=1)
        setattr(tree, 'btag_avg', self.btag_avg)

        sample_particles = []
        for j in range(6):
            particle = Particle({
                'pt' : pt[:,j],
                'eta' : eta[:,j],
                'phi' : phi[:,j],
                'm' : m[:,j],
                'btag' : btag[:,j],
                'sig_id' : sig_id[:,j],
                'h_id': h_id[:,j]
                }
            )
            sample_particles.append(particle)

        HX_b1 = {'pt':sample_particles[0].pt,'eta':sample_particles[0].eta,'phi':sample_particles[0].phi,'m':sample_particles[0].m,'btag':sample_particles[0].btag,'sig_id':sample_particles[0].sig_id,'h_id':sample_particles[0].h_id}
        HX_b2 = {'pt':sample_particles[1].pt,'eta':sample_particles[1].eta,'phi':sample_particles[1].phi,'m':sample_particles[1].m,'btag':sample_particles[1].btag,'sig_id':sample_particles[1].sig_id,'h_id':sample_particles[1].h_id}
        H1_b1 = {'pt':sample_particles[2].pt,'eta':sample_particles[2].eta,'phi':sample_particles[2].phi,'m':sample_particles[2].m,'btag':sample_particles[2].btag,'sig_id':sample_particles[2].sig_id,'h_id':sample_particles[2].h_id}
        H1_b2 = {'pt':sample_particles[3].pt,'eta':sample_particles[3].eta,'phi':sample_particles[3].phi,'m':sample_particles[3].m,'btag':sample_particles[3].btag,'sig_id':sample_particles[3].sig_id,'h_id':sample_particles[3].h_id}
        H2_b1 = {'pt':sample_particles[4].pt,'eta':sample_particles[4].eta,'phi':sample_particles[4].phi,'m':sample_particles[4].m,'btag':sample_particles[4].btag,'sig_id':sample_particles[4].sig_id,'h_id':sample_particles[4].h_id}
        H2_b2 = {'pt':sample_particles[5].pt,'eta':sample_particles[5].eta,'phi':sample_particles[5].phi,'m':sample_particles[5].m,'btag':sample_particles[5].btag,'sig_id':sample_particles[5].sig_id,'h_id':sample_particles[5].h_id}

        self.HX = Higgs(HX_b1, HX_b2)
        setattr(tree, 'HX', self.HX)

        H1 = Higgs(H1_b1, H1_b2)
        H2 = Higgs(H2_b1, H2_b2)

        assert ak.all(self.HX.b1.pt >= self.HX.b2.pt)
        assert ak.all(H1.b1.pt >= H1.b2.pt)
        assert ak.all(H2.b1.pt >= H2.b2.pt)

        self.Y = Y(H1, H2)
        setattr(tree, 'Y', self.Y)

        self.H1 = self.Y.H1
        self.H2 = self.Y.H2
        setattr(tree, 'H1', self.H1)
        setattr(tree, 'H2', self.H2)

        assert ak.all(self.H1.pt >= self.H2.pt)

        self.X = self.HX + self.H1 + self.H2
        setattr(tree, 'X', self.X)

        self.higgs_bjet_sig_id = np.column_stack((
            self.HX.b1.sig_id.to_numpy(),
            self.HX.b2.sig_id.to_numpy(),
            self.H1.b1.sig_id.to_numpy(),
            self.H1.b2.sig_id.to_numpy(),
            self.H2.b1.sig_id.to_numpy(),
            self.H2.b2.sig_id.to_numpy(),
        ))

        self.higgs_bjet_h_id = np.column_stack((
            self.HX.b1.h_id.to_numpy(),
            self.HX.b2.h_id.to_numpy(),
            self.H1.b1.h_id.to_numpy(),
            self.H1.b2.h_id.to_numpy(),
            self.H2.b1.h_id.to_numpy(),
            self.H2.b2.h_id.to_numpy(),
        ))

        self.feyn_resolved_mask = ak.all(self.higgs_bjet_sig_id > -1, axis=1)
        self.feyn_resolved_h_mask = ak.all(self.higgs_bjet_h_id > 0, axis=1)

        hx_possible = ak.sum(self.higgs_bjet_h_id == 1, axis=1) == 2
        h1_possible = ak.sum(self.higgs_bjet_h_id == 2, axis=1) == 2
        h2_possible = ak.sum(self.higgs_bjet_h_id == 3, axis=1) == 2
        self.n_h_possible = hx_possible*1 + h1_possible*1 + h2_possible*1

        # efficiency without taking into account to which higgs the pair was assigned
        # only worried about pairing together two jets from any higgs boson
        hx_correct = (self.HX.b1.h_id == self.HX.b2.h_id) & (self.HX.b1.h_id > 0)*1
        h1_correct = (self.H1.b1.h_id == self.H1.b2.h_id) & (self.H1.b1.h_id > 0)*1
        h2_correct = (self.H2.b1.h_id == self.H2.b2.h_id) & (self.H2.b1.h_id > 0)*1
        self.n_H_paired_correct = hx_correct + h1_correct + h2_correct

    def copy_attributes(self, dest_cls):
        # but not methods
        for attr_name, attr_value in vars(self).items():
            if not callable(attr_value): setattr(dest_cls, attr_name, attr_value)