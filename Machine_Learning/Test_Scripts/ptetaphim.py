from getFile import getFile
import uproot
import uproot_methods

class jet:
    
    def __init__(self, pt, eta, phi, m):
        
        self.p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(pt, eta, phi, m)
        
        self.pt  = self.p4.pt
        self.eta = self.p4.eta
        self.phi = self.p4.phi
        self.m   = self.p4.mass
        
class jet_from_dict:
    
    def __init__(self, part_dict):
        
        self.p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(part_dict['pt'], part_dict['eta'], part_dict['phi'], part_dict['m'])
        
        self.pt  = self.p4.pt
        self.eta = self.p4.eta
        self.phi = self.p4.phi
        self.m   = self.p4.mass
        
        
class dijet:
    
    def __init__(self, p4s):
        
        if len(p4s) == 2:
            self.p4 = p4s[0] + p4s[1]

        self.pt  = self.p4.pt
        self.eta = self.p4.eta
        self.phi = self.p4.phi
        self.m   = self.p4.mass
        
        
class multijet:
    
    def __init__(self, p4s):
        
        self.p4 = p4s[0] + p4s[1]
        
        for i in range(2,len(p4s)):
            self.p4 = self.p4 + p4[i]
            
        self.pt  = self.p4.pt
        self.eta = self.p4.eta
        self.phi = self.p4.phi
        self.m   = self.p4.mass