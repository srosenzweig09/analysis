import awkward as ak
import numpy as np
np.seterr(all="ignore")
import vector
vector.register_awkward()
import sys

import inspect

# def particle_from_tree(tree, gen=False):
#     if gen:
        
        

class Particle():
    # def __init__(self, tree=None, particle_name=None, particle=None, kin_dict=None):
    def __init__(self, particle, particle_name=None):
        from utils.analysis import SixB, Data
        """A class much like the TLorentzVector objects in the vector module but with more customizability for my analysis
        """
        # if tree is not None and particle_name is not None:
        # elif particle is not None:
        if isinstance(particle, vector.MomentumNumpy4D):
            self.initialize_from_particle(particle)
        elif isinstance(particle, dict):
            self.initialize_from_kinematics(particle)
        else:
            self.initialize_from_tree(particle, particle_name)

        self.P4 = self.get_vector()
        self.theta = self.P4.theta
        self.costheta = np.cos(self.theta)

    def initialize_from_kinematics(self, kin_dict):
        for key,val in kin_dict.items():
            setattr(self, key, val)

    def initialize_from_tree(self, tree, particle_name):
        try: self.pt = getattr(tree, particle_name + '_ptRegressed')
        except: self.pt = getattr(tree, particle_name + '_pt')
        self.eta = getattr(tree, particle_name + '_eta')
        self.phi = getattr(tree, particle_name + '_phi')
        self.m = getattr(tree, particle_name + '_m')
        try:
            self.btag = getattr(tree, particle_name + '_btag')
            self.h_id = getattr(tree, particle_name + '_genHflag')
            # self.h_id = (self.sig_id + 2) // 2
        except: pass

    def initialize_from_particle(self, particle):
        self.pt = particle.pt
        self.eta = particle.eta
        self.phi = particle.phi
        self.m = particle.m
        try: self.btag = particle.btag
        except: pass

    def get_vector(self):
        p4 = vector.arr({
            "pt"  : self.pt,
            "eta" : self.eta,
            "phi" : self.phi,
            "m"   : self.m
        })
        return p4

    def set_attr(self, key, val):
        setattr(self, key, val)

    def __add__(self, another_particle):
        particle1 = self.P4
        particle2 = another_particle.P4
        parent = particle1 + particle2
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

        H1_b1_dict = {}
        H1_b2_dict = {}
        H2_b1_dict = {}
        H2_b2_dict = {}

        for key in H1.b1.__dict__.keys():
            try: 
                H1_b1_dict[key] = ak.where(H1.pt >= H2.pt, getattr(H1.b1, key), getattr(H2.b1, key))
                H1_b2_dict[key] = ak.where(H1.pt >= H2.pt, getattr(H1.b2, key), getattr(H2.b2, key))
                H2_b1_dict[key] = ak.where(H1.pt <  H2.pt, getattr(H1.b1, key), getattr(H2.b1, key))
                H2_b2_dict[key] = ak.where(H1.pt <  H2.pt, getattr(H1.b2, key), getattr(H2.b2, key))
            except: pass

        self.H1_b1 = Particle(H1_b1_dict)
        self.H1_b2 = Particle(H1_b2_dict)
        self.H2_b1 = Particle(H2_b1_dict)
        self.H2_b2 = Particle(H2_b2_dict)

        self.H1 = self.H1_b1 + self.H1_b2
        self.H2 = self.H2_b1 + self.H2_b2

        setattr(self.H1, 'b1', self.H1_b1)
        setattr(self.H1, 'b2', self.H1_b2)
        setattr(self.H2, 'b1', self.H2_b1)
        setattr(self.H2, 'b2', self.H2_b2)

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