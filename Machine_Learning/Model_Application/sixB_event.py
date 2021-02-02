import uproot
import uproot_methods
import awkward
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
# import matplotlib as mpl

plt.rcParams.update({'font.size': 18})

class sixB_event():
    def calcDeltas(self, phi, eta):
        """
        This function is used to calculate Delta phi and Delta eta between two particles originating
        from the same parent.
        
        Keyword arguments:
        phi -- a two-element list of phi1 and phi2
        eta -- a two-element list of eta1 and eta2
        """
        Dphi = phi[0] - phi[1]
        Dphi = np.where(Dphi > +np.pi, Dphi - 2*np.pi, Dphi)
        Dphi = np.where(Dphi < -np.pi, Dphi + 2*np.pi, Dphi)
        
        Deta = eta[0] - eta[1]
        
        DR = np.sqrt(Dphi**2 + Deta**2)
        
        return Dphi, Deta, DR
        
    def __init__(self, mx,  my):

        # We'll be saving all of the plots to the same location so let's make this simple.
        self.save_loc = 'gen_plots/X_YH_HHH_6b_MX' + str(mx) + '_MY' + str(my) + '/'

        # Let's see where we're saving everything.
        print(self.save_loc)

        filename = 'Mass_Pair_ROOT_files/X_YH_HHH_6b_MX' + str(mx) + '_MY' + str(my) + '.root'
        
        f = uproot.open(filename)
        tree = f['sixbntuplizer/sixBtree']
        branches = tree.arrays(namedecode='utf-8')

        self.keys = branches.keys()
        
        self.table = awkward.Table(branches)
        
        self.gen_jet_pt = self.table['gen_jet_pt']
        self.gen_jet_eta = self.table['gen_jet_eta']
        self.gen_jet_phi = self.table['gen_jet_phi']
        self.gen_jet_m = self.table['gen_jet_m']
        
        self.X_m   = self.table['gen_lc_X_m']
        self.X_pt  = self.table['gen_lc_X_pt']
        self.X_eta = self.table['gen_lc_X_eta']
        self.X_phi = self.table['gen_lc_X_phi']
        
        self.Y_m     = self.table['gen_lc_Y_m']
        self.Y_pt    = self.table['gen_lc_Y_pt']
        self.Y_eta   = self.table['gen_lc_Y_eta']
        self.Y_phi   = self.table['gen_lc_Y_phi']

        self.H_m    = self.table['gen_lc_HX_m']
        self.H_pt   = self.table['gen_lc_HX_pt']
        self.H_eta  = self.table['gen_lc_HX_eta']
        self.H_phi  = self.table['gen_lc_HX_phi']
        
        self.H_b1_pt  = self.table['gen_HX_b1_pt']
        self.H_b2_pt  = self.table['gen_HX_b2_pt']
        self.H_b1_eta = self.table['gen_HX_b1_eta']
        self.H_b2_eta = self.table['gen_HX_b2_eta']
        self.H_b1_phi = self.table['gen_HX_b1_phi']
        self.H_b2_phi = self.table['gen_HX_b2_phi']
        self.H_b1_m   = self.table['gen_HX_b1_m']
        self.H_b2_m   = self.table['gen_HX_b2_m']
        
        self.H_Deltas = self.calcDeltas([self.H_b1_phi, self.H_b2_phi], [self.H_b1_eta, self.H_b2_eta])

        self.H1_m   = self.table['gen_lc_HY1_m']
        self.H1_pt  = self.table['gen_lc_HY1_pt']
        self.H1_eta = self.table['gen_lc_HY1_eta']
        self.H1_phi = self.table['gen_lc_HY1_phi']
        
        self.H1_b1_pt  = self.table['gen_HY1_b1_pt']
        self.H1_b2_pt  = self.table['gen_HY1_b2_pt']
        self.H1_b1_eta = self.table['gen_HY1_b1_eta']
        self.H1_b2_eta = self.table['gen_HY1_b2_eta']
        self.H1_b1_phi = self.table['gen_HY1_b1_phi']
        self.H1_b2_phi = self.table['gen_HY1_b2_phi']
        self.H1_b1_m   = self.table['gen_HY1_b1_m']
        self.H1_b2_m   = self.table['gen_HY1_b2_m']
        
        self.H1_Deltas = self.calcDeltas([self.H1_b1_phi, self.H1_b2_phi], [self.H1_b1_eta, self.H1_b2_eta])

        self.H2_m   = self.table['gen_lc_HY2_m']
        self.H2_pt  = self.table['gen_lc_HY2_pt']
        self.H2_eta = self.table['gen_lc_HY2_eta']
        self.H2_phi = self.table['gen_lc_HY2_phi']
        
        self.H2_b1_pt  = self.table['gen_HY2_b1_pt']
        self.H2_b2_pt  = self.table['gen_HY2_b2_pt']
        self.H2_b1_eta = self.table['gen_HY2_b1_eta']
        self.H2_b2_eta = self.table['gen_HY2_b2_eta']
        self.H2_b1_phi = self.table['gen_HY2_b1_phi']
        self.H2_b2_phi = self.table['gen_HY2_b2_phi']
        self.H2_b1_m   = self.table['gen_HY2_b1_m']
        self.H2_b2_m   = self.table['gen_HY2_b2_m']
        
        self.H2_Deltas = self.calcDeltas([self.H2_b1_phi, self.H2_b2_phi], [self.H2_b1_eta, self.H2_b2_eta])
        
        self.Y_H_Deltas = self.calcDeltas([self.table['gen_lc_Y_phi'], self.table['gen_lc_HX_phi']],
                                      [self.table['gen_lc_Y_eta'], self.table['gen_lc_HX_eta']])
        self.deltaPhi_Y_H = self.Y_H_Deltas[0]
        self.deltaEta_Y_H = self.Y_H_Deltas[1]
        self.deltaR_Y_H = self.Y_H_Deltas[2]
        
        self.H1_H2_Deltas = self.calcDeltas([self.table['gen_lc_HY1_phi'],
                                               self.table['gen_lc_HY2_phi']],
                                              [self.table['gen_lc_HY1_eta'],
                                               self.table['gen_lc_HY2_eta']])
        self.deltaPhi_H1_H2 = self.H1_H2_Deltas[0]
        self.deltaEta_H1_H2 = self.H1_H2_Deltas[1]
        self.deltaR_H1_H2 = self.H1_H2_Deltas[2]
        
        self.H1_H_Deltas = self.calcDeltas([self.table['gen_lc_HY1_phi'], self.table['gen_lc_HX_phi']],
                                      [self.table['gen_lc_HY1_eta'], self.table['gen_lc_HX_eta']])
        self.deltaPhi_H1_H = self.H1_H_Deltas[0]
        self.deltaEta_H1_H = self.H1_H_Deltas[1]
        self.deltaR_H1_H = self.H1_H_Deltas[2]
        
        self.H_H2_Deltas = self.calcDeltas([self.table['gen_lc_HX_phi'], self.table['gen_lc_HY2_phi']],
                                      [self.table['gen_lc_HX_eta'], self.table['gen_lc_HY2_eta']])
        self.deltaPhi_H_H2 = self.H_H2_Deltas[0]
        self.deltaEta_H_H2 = self.H_H2_Deltas[1]
        self.deltaR_H_H2 = self.H_H2_Deltas[2]
        
        self.b_quark_jetId = np.vstack((self.table['gen_HX_b1_genjetIdx'],
                                        self.table['gen_HX_b2_genjetIdx'],
                                        self.table['gen_HY1_b1_genjetIdx'],
                                        self.table['gen_HY1_b2_genjetIdx'],
                                        self.table['gen_HY2_b1_genjetIdx'],
                                        self.table['gen_HY2_b2_genjetIdx']))
        
        i = 0
        nJet = []
        self.b_quark_pt = np.vstack((self.table['gen_HX_b1_pt'], self.table['gen_HX_b2_pt'],
                             self.table['gen_HY1_b1_pt'], self.table['gen_HY1_b2_pt'],
                             self.table['gen_HY2_b1_pt'], self.table['gen_HY2_b2_pt']))
        
        self.b_quark_eta = np.vstack((self.table['gen_HX_b1_eta'], self.table['gen_HX_b2_eta'],
                             self.table['gen_HY1_b1_eta'], self.table['gen_HY1_b2_eta'],
                             self.table['gen_HY2_b1_eta'], self.table['gen_HY2_b2_eta']))
        
        
        particle_dict = {0:'gen_HX_b1', 1:'gen_HX_b2', 2:'gen_HY1_b1', 3:'gen_HY1_b2', 
                         4:'gen_HY2_b1', 5:'gen_HY2_b2'}
        
        sorted_jetId = []
        sorted_b_pt = []
        order_b_pt = []
        sorted_b_eta = []
        order_b_phi = []
        
        for pt, eta in zip(self.table['gen_jet_pt'], self.table['gen_jet_eta']):
            nJet.append(len(pt))
            
            evt_pt = self.b_quark_pt[:,i]
            lin_arr = np.arange(0,6,1)
            
            arr1inds = evt_pt.argsort()
            sorted_b_pt.append(evt_pt[arr1inds[::-1]])
            sorted_b_eta.append(self.b_quark_eta[:,i][arr1inds[::-1]])
            sorted_jetId.append(self.b_quark_jetId[:,i][arr1inds[::-1]])
            temp = lin_arr[arr1inds[::-1]]
            temp_empty = []
            for item in temp:
                temp_empty.append(particle_dict[item])
            order_b_pt.append(temp_empty)
            
            i += 1
            
        self.sorted_b_pt = np.array(sorted_b_pt)
        self.sorted_b_eta = np.array(sorted_b_eta)
        self.sorted_b_jetId = np.array(sorted_jetId)
        self.order_b_pt = np.array(order_b_pt)
        self.nJet = np.array(nJet)


