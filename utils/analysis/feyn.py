from configparser import ConfigParser
import json, uproot
import numpy as np
import awkward as ak
from utils.analysis.particle import Particle, Higgs, Y

def getMassDict(path):
    with open(f"{path}/samples.json", 'r') as file:
        mass_dict = json.load(file)
    return mass_dict

class Model():

    @staticmethod
    def read_feynnet_cfg(cfg="config/feynnet.cfg"):
        config = ConfigParser()
        config.read(cfg)

        model_version = config['feynnet']['version']
        model_name = config['feynnet']['name']
        model_path = config['feynnet']['path'].replace('version', model_name)
        return model_version, model_name, model_path

    @staticmethod
    def get_savepath(cfg="config/feynnet.cfg"):
        model_version, model_name, model_path = Model.read_feynnet_cfg(cfg)
        return f"plots/feynnet/{model_name}"

    def __init__(self, tree, config="config/feynnet.cfg"):
        model_version, model_name, model_path = self.read_feynnet_cfg(config)

        if model_version == 'new': self.init_new_model(tree, model_path)
        elif model_version == 'old': self.init_old_model(tree, model_path)
        else: raise ValueError(f"Model type '{model_version}' not recognized")

        self.version = model_version
        self.model_name = model_name
        self.savepath = f"plots/feynnet/{self.model_name}"

        self.init_particles(tree)
        self.copy_attributes(tree)

    def init_new_model(self, tree, path):
        mass_dict = getMassDict(path)
        model_path = mass_dict[tree.filepath]
        with uproot.open(model_path) as f:
            t = f['Events']
            self.combos = t['sorted_j_assignments'].array()
            self.ranks = t['sorted_rank'].array()
            maxcomb = ak.firsts(t['sorted_j_assignments'].array())
            sorted_rank = ak.firsts(t['sorted_rank'].array())
        self.combo = ak.from_regular(maxcomb)
        self.rank = sorted_rank
        self.version = 'new'

    def init_old_model(self, tree, path):
        path = f"{path}/{tree.year_long}/{tree.filename}.root"
        with uproot.open(path) as f:
            f = f['Events']
            maxcomb = f['max_comb'].array(library='np')

        combo = maxcomb.astype(int)
        self.combo = ak.from_regular(combo)
        self.version = 'old'

    def init_particles(self, tree, combo=0):
        if combo == 0: combo = self.combo
        else: combo = ak.from_regular(self.combos[:,combo])

        btag_mask = ak.argsort(tree.jet_btag, axis=1, ascending=False) < 6

        pt = tree.jet_ptRegressed[btag_mask][combo]
        phi = tree.jet_phi[btag_mask][combo]
        eta = tree.jet_eta[btag_mask][combo]
        m = tree.jet_mRegressed[btag_mask][combo]
        btag = tree.jet_btag[btag_mask][combo]
        sig_id = tree.jet_signalId[btag_mask][combo]
        h_id = (tree.jet_signalId[btag_mask][combo] + 2) // 2

        self.btag_avg = ak.mean(btag, axis=1)

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

        H1 = Higgs(HX_b1, HX_b2)
        H2 = Higgs(H1_b1, H1_b2)
        H3 = Higgs(H2_b1, H2_b2)
        self.Y = Y(H2, H3) # should not be used

        if self.version == 'new':
            from utils.analysis.particle import OrderedSixB
            self.pt_ordered_h = OrderedSixB(H1, H2, H3)

            self.H1 = self.pt_ordered_h.H1
            self.H2 = self.pt_ordered_h.H2
            self.H3 = self.pt_ordered_h.H3

        if self.version == 'old':
            self.H1 = H1
            self.H2, self.H3 = Particle.swap_objects(H2, H3, H2.pt < H3.pt)
            assert ak.all(self.H2.pt >= self.H3.pt)
            self.Y = Y(H2, H3)

        self.HY1 = self.Y.H1
        self.HY2 = self.Y.H2

        # assert ak.all(self.H1.pt >= self.H2.pt)

        # self.X = self.HX + self.H1 + self.H2
        self.X = self.H1 + self.H2 + self.H3
        # setattr(tree, 'X', self.X)

        if tree._is_signal:
            self.higgs_bjet_sig_id = np.column_stack((
                self.H1.b1.sig_id.to_numpy(),
                self.H1.b2.sig_id.to_numpy(),
                self.H2.b1.sig_id.to_numpy(),
                self.H2.b2.sig_id.to_numpy(),
                self.H3.b1.sig_id.to_numpy(),
                self.H3.b2.sig_id.to_numpy(),
            ))

            self.higgs_bjet_h_id = np.column_stack((
                self.H1.b1.h_id.to_numpy(),
                self.H1.b2.h_id.to_numpy(),
                self.H2.b1.h_id.to_numpy(),
                self.H2.b2.h_id.to_numpy(),
                self.H3.b1.h_id.to_numpy(),
                self.H3.b2.h_id.to_numpy(),
            ))

            self.feyn_resolved_mask = ak.all(self.higgs_bjet_sig_id > -1, axis=1)
            self.feyn_resolved_h_mask = ak.all(self.higgs_bjet_h_id > 0, axis=1)

            hx_possible = ak.sum(self.higgs_bjet_h_id == 1, axis=1) == 2
            h1_possible = ak.sum(self.higgs_bjet_h_id == 2, axis=1) == 2
            h2_possible = ak.sum(self.higgs_bjet_h_id == 3, axis=1) == 2
            self.n_h_possible = hx_possible*1 + h1_possible*1 + h2_possible*1

            # efficiency without taking into account to which higgs the pair was assigned
            # only worried about pairing together two jets from any higgs boson
            hx_correct = (self.H1.b1.h_id == self.H1.b2.h_id) & (self.H1.b1.h_id > 0)
            h1_correct = (self.H2.b1.h_id == self.H2.b2.h_id) & (self.H2.b1.h_id > 0)
            h2_correct = (self.H3.b1.h_id == self.H3.b2.h_id) & (self.H3.b1.h_id > 0)
            self.n_h_found = hx_correct*1 + h1_correct*1 + h2_correct*1

    def copy_attributes(self, dest_cls):
        # but not methods
        for attr_name, attr_value in vars(self).items():
            if not callable(attr_value): setattr(dest_cls, attr_name, attr_value)