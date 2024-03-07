import awkward as ak
import numpy as np
np.seterr(all="ignore")
import vector
vector.register_awkward()
# import sys
# import inspect


class Particle():


    @staticmethod
    def swap_arrays(this, that, swap):
        _this = ak.where(swap, that, this)
        _that = ak.where(swap, this, that)
        return _this, _that

    @staticmethod
    def swap_objects(this, that, swap):
        this_fields = [ key for key, value in vars(this).items() if isinstance(value, ak.Array)]
        that_fields = [ key for key, value in vars(that).items() if isinstance(value, ak.Array)]

        assert set(this_fields) == set(that_fields), f"Fields of this and that are not the same: {this_fields} != {that_fields}"

        for field in this_fields:
            this_field, that_field = Particle.swap_arrays(getattr(this, field), getattr(that, field), swap)
            setattr(this, field, this_field)
            setattr(that, field, that_field)

        return this, that

    def __init__(self, particle, particle_name=None):
        from utils.analysis import SixB, Data
        """A class much like the TLorentzVector objects in the vector module but with more customizability for my analysis
        """
        # if tree is not None and particle_name is not None:
        # elif particle is not None:
        MomentumArray4D = vector.backends.awkward.MomentumArray4D
        if isinstance(particle, vector.MomentumNumpy4D) or isinstance(particle, MomentumArray4D):
            self.initialize_from_particle(particle)
        elif isinstance(particle, dict):
            self.initialize_from_kinematics(particle)
        else:
            # print("Checkpoint 2")
            self.initialize_from_tree(particle, particle_name)


        self.P4 = self.get_vector()
        # self.old_P4 = self.get_old_vector()
        self.theta = self.P4.theta
        self.costheta = np.cos(self.theta)

    def initialize_from_kinematics(self, kin_dict):
        btag_flag = False
        for key,val in kin_dict.items():
            setattr(self, key, val)
            # if key == 'signalId': btag_flag = True
        # if btag_flag: self.set_h_id()

    def initialize_from_tree(self, tree, particle_name):
        try: 
            self.pt = getattr(tree, particle_name + '_ptRegressed')
        except: 
            self.pt = getattr(tree, particle_name + '_pt')
        self.eta = getattr(tree, particle_name + '_eta')
        self.phi = getattr(tree, particle_name + '_phi')
        self.m = getattr(tree, particle_name + '_m')
        try:
            self.btag = getattr(tree, particle_name + '_btag')
            self.h_id = getattr(tree, particle_name + '_genHflag')
            # self.h_id = (self.sig_id + 2) // 2
        except: pass
        # print("Checkpoint 3")

    def set_h_id(self):
        self.h_id = (self.sig_id + 2) // 2

    def initialize_from_particle(self, particle):
        self.pt = particle.pt
        self.eta = particle.eta
        self.phi = particle.phi
        self.m = particle.m
        try: self.btag = particle.btag
        except: pass

    def get_vector(self):
        p4 = ak.zip({
            "pt"  : self.pt,
            "eta" : self.eta,
            "phi" : self.phi,
            "m"   : self.m
        }, with_name='Momentum4D')
        # print(type(p4))
        return p4

    def get_kin_dict(self):
        return {
            'pt': self.pt,
            'eta': self.eta,
            'phi': self.phi,
            'm': self.m,
            'btag' : self.btag,
            'h_id' : self.h_id
        }

    def set_attr(self, key, val):
        setattr(self, key, val)

    def __add__(self, another_particle):
        particle1 = self.P4
        particle2 = another_particle.P4
        parent = particle1 + particle2
        # print(type(parent))
        parent = Particle(particle=parent)
        parent.set_attr('dr', particle1.deltaR(particle2))
        return parent
    
    def boost(self, another_particle):
        return self.P4.boost(-another_particle.P4)

    def deltaEta(self, another_particle):
        return self.P4.deltaeta(another_particle.P4)

    def deltaPhi(self, another_particle):
        return self.P4.deltaphi(another_particle.P4)

    def deltaR(self, another_particle):
        return self.P4.deltaR(another_particle.P4)

class Higgs():
    def __init__(self, kin_dict1, kin_dict2):

        assert 'pt' in kin_dict1.keys() and 'pt' in kin_dict2.keys()
        assert kin_dict1.keys() == kin_dict2.keys()

        b1_dict = {}
        b2_dict = {}

        for key in kin_dict1.keys():
            b1_dict[key] = ak.where(kin_dict1['pt'] >= kin_dict2['pt'], kin_dict1[key], kin_dict2[key])
            b2_dict[key] = ak.where(kin_dict1['pt'] <  kin_dict2['pt'], kin_dict1[key], kin_dict2[key])

        self.b1 = Particle(b1_dict)
        self.b2 = Particle(b2_dict)

        H = self.b1 + self.b2
        
        self.h_id = np.where(self.b1.h_id == self.b2.h_id, self.b1.h_id, 0)

        self.pt = H.pt
        self.eta = H.eta
        self.phi = H.phi
        self.m = H.m

        self.P4 = H.P4

        self.dr = self.b1.deltaR(self.b2)
        self.theta = self.P4.theta
        self.costheta = abs(np.cos(self.theta))

    def __add__(self, another_particle):
        particle1 = self.P4
        particle2 = another_particle.P4
        parent = particle1 + particle2
        return Particle(particle=parent)
    
    def boost(self, another_particle):
        return self.P4.boost(-another_particle.P4)

    def deltaEta(self, another_particle):
        return self.P4.deltaeta(another_particle.P4)

    def deltaPhi(self, another_particle):
        return self.P4.deltaphi(another_particle.P4)

    def deltaR(self, another_particle):
        return self.P4.deltaR(another_particle.P4)

class Y():
    def __init__(self, H1, H2):

        swap_h1_h2 = H1.pt < H2.pt
        H1, H2 = Particle.swap_objects(H1, H2, swap_h1_h2)
        assert ak.all(H1.pt >= H2.pt)

        self.H1 = H1
        self.H2 = H2

        assert ak.all(self.H1.pt >= self.H2.pt)

        Y = self.H1 + self.H2

        self.H1.dr = H1.dr
        self.H2.dr = H2.dr

        self.P4 = Y.P4
        self.pt = Y.pt
        self.eta = Y.eta
        self.phi = Y.phi
        self.m = Y.m
        
    def __add__(self, another_particle):
        particle1 = self.P4
        particle2 = another_particle.P4
        parent = particle1 + particle2
        return Particle(particle=parent)
    
    def boost(self, another_particle):
        return self.P4.boost(-another_particle.P4)

    def deltaEta(self, another_particle):
        return self.P4.deltaeta(another_particle.P4)

    def deltaPhi(self, another_particle):
        return self.P4.deltaphi(another_particle.P4)

    def deltaR(self, another_particle):
        return self.P4.deltaR(another_particle.P4)

class OrderedSixB():

    def __init__(self, H1, H2, H3):

        swap_h1_h2 = H1.pt < H2.pt
        H1, H2 = Particle.swap_objects(H1, H2, swap_h1_h2)
        assert ak.all(H1.pt >= H2.pt)

        swap_h1_h3 = H1.pt < H3.pt
        H1, H3 = Particle.swap_objects(H1, H3, swap_h1_h3)
        assert ak.all(H1.pt >= H3.pt)

        swap_h2_h3 = H2.pt < H3.pt
        H2, H3 = Particle.swap_objects(H2, H3, swap_h2_h3)
        assert ak.all(H2.pt >= H3.pt)
        assert ak.all(H1.pt >= H2.pt)

        self.H1 = H1
        self.H2 = H2
        self.H3 = H3
