from myuproot import get_uproot_Table
from uproot3_methods import TLorentzVectorArray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from kinematics import calcDeltaR
from consistent_plots import hist

reco_filename = f'/eos/user/s/srosenzw/SWAN_projects/sixB/Analysis_6b/Data_Preparation/NMSSM_XYH_YToHH_6b_MX_700_MY_400_accstudies.root'
table =  get_uproot_Table(reco_filename, 'sixBtree')
nevt = table._length()


pt1 = table['gen_HX_b1_pt']
eta1 = table['gen_HX_b1_eta']
phi1 = table['gen_HX_b1_phi']
m1 = table['gen_HX_b1_m']
pt2 = table['gen_HX_b2_pt']
eta2 = table['gen_HX_b2_eta']
phi2 = table['gen_HX_b2_phi']
m2 = table['gen_HX_b2_m']

print(m1)
print(m2)

b1_p4 = TLorentzVectorArray.from_ptetaphim(pt1, eta1, phi1, m1)
b2_p4 = TLorentzVectorArray.from_ptetaphim(pt2, eta2, phi2, m2)

uproot_dR = b1_p4.delta_r(b2_p4)
suzanne_dR = calcDeltaR(eta1, eta2, phi1, phi2)

widths = [1]
heights = [4, 1]
gs_kw = dict(width_ratios=widths, height_ratios=heights)
fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw=gs_kw)

dR_bins = np.linspace(0,7,100)
ax  = axs[0]
n_up, edges, _ = hist(ax, uproot_dR, bins=dR_bins, label='Uproot')
n_suz, edges, _ = hist(ax, suzanne_dR, bins=dR_bins, label='My Function')
ax.set_xlabel(r'$\Delta R_{H,bb}$')
ax.legend()

ax  = axs[1]
ratio = n_up / n_suz
x = (dR_bins[1:] + dR_bins[:-1])/2
ax.plot(x, ratio)

fig.savefig('DeltaR_comparison')