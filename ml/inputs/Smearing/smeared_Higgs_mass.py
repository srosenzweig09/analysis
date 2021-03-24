from uproot3 import open
from uproot3_methods import TLorentzVectorArray
from awkward0 import Table
import numpy as np
import matplotlib.pyplot as plt
# import mplhep as hep
# hep.set_style(hep.style.CMS)
# plt.style.use(hep.style.CMS)
from logger import info

def f_Gauss(pt):
    """Randomly sample a Gaussian with mu = 0 and sigma = 0.1
    """
    return np.random.normal(0, 0.1, len(pt))

filename = '/eos/user/s/srosenzw/SWAN_projects/sixB/Signal_Exploration/Mass_Pair_ROOT_files/X_YH_HHH_6b_MX700_MY400.root'
examples = np.load('MX700_MY400_classifier_allpairs_dR_presel_smeared.npz')

info("Opening signal event ROOT file.")
f = open(filename)
t = f['sixbntuplizer/sixBtree']
b = t.arrays(namedecode='utf-8')
t = Table(b)

HX_b1  = {'pt':t['gen_HX_b1_pt'],  'eta':t['gen_HX_b1_eta'],  'phi':t['gen_HX_b1_phi'],  'm':t['gen_HX_b1_m']}
HX_b2  = {'pt':t['gen_HX_b2_pt'],  'eta':t['gen_HX_b2_eta'],  'phi':t['gen_HX_b2_phi'],  'm':t['gen_HX_b2_m']}
HX     = {'b1':HX_b1, 'b2':HX_b2}

HX_b1_pt_smeared = (1+f_Gauss(HX['b1']['pt']))*HX['b1']['pt']
HX_b2_pt_smeared = (1+f_Gauss(HX['b2']['pt']))*HX['b2']['pt']

HY1_b1 = {'pt':t['gen_HY1_b1_pt'], 'eta':t['gen_HY1_b1_eta'], 'phi':t['gen_HY1_b1_phi'], 'm':t['gen_HY1_b1_m']}
HY1_b2 = {'pt':t['gen_HY1_b2_pt'], 'eta':t['gen_HY1_b2_eta'], 'phi':t['gen_HY1_b2_phi'], 'm':t['gen_HY1_b2_m']}
HY1    = {'b1':HY1_b1, 'b2':HY1_b2}

HY1_b1_pt_smeared = (1+f_Gauss(HY1['b1']['pt']))*HY1['b1']['pt']
HY1_b2_pt_smeared = (1+f_Gauss(HY1['b2']['pt']))*HY1['b2']['pt']

HY2_b1 = {'pt':t['gen_HY2_b1_pt'], 'eta':t['gen_HY2_b1_eta'], 'phi':t['gen_HY2_b1_phi'], 'm':t['gen_HY2_b1_m']}
HY2_b2 = {'pt':t['gen_HY2_b2_pt'], 'eta':t['gen_HY2_b2_eta'], 'phi':t['gen_HY2_b2_phi'], 'm':t['gen_HY2_b2_m']}
HY2    = {'b1':HY2_b1, 'b2':HY2_b2}


HY2_b1_pt_smeared = (1+f_Gauss(HY2['b1']['pt']))*HY2['b1']['pt']
HY2_b2_pt_smeared = (1+f_Gauss(HY2['b2']['pt']))*HY2['b2']['pt']

HXb1_p4 = TLorentzVectorArray.from_ptetaphim(HX['b1']['pt'], HX['b1']['eta'], HX['b1']['phi'], HX['b1']['m'])
HXb2_p4 = TLorentzVectorArray.from_ptetaphim(HX['b2']['pt'], HX['b2']['eta'], HX['b2']['phi'], HX['b2']['m'])
HXb1_p4_s = TLorentzVectorArray.from_ptetaphim(HX_b1_pt_smeared, HX['b1']['eta'], HX['b1']['phi'], HX['b1']['m'])
HXb2_p4_s = TLorentzVectorArray.from_ptetaphim(HX_b2_pt_smeared, HX['b2']['eta'], HX['b2']['phi'], HX['b2']['m'])
HX_m = (HXb1_p4 + HXb2_p4).mass
HX_m_s = (HXb1_p4_s + HXb2_p4_s).mass

HY1b1_p4 = TLorentzVectorArray.from_ptetaphim(HY1['b1']['pt'], HY1['b1']['eta'], HY1['b1']['phi'], HY1['b1']['m'])
HY1b2_p4 = TLorentzVectorArray.from_ptetaphim(HY1['b2']['pt'], HY1['b2']['eta'], HY1['b2']['phi'], HY1['b2']['m'])
HY1b1_p4_s = TLorentzVectorArray.from_ptetaphim(HY1_b1_pt_smeared, HY1['b1']['eta'], HY1['b1']['phi'], HY1['b1']['m'])
HY1b2_p4_s = TLorentzVectorArray.from_ptetaphim(HY1_b2_pt_smeared, HY1['b2']['eta'], HY1['b2']['phi'], HY1['b2']['m'])
HY1_m = (HY1b1_p4 + HY1b2_p4).mass
HY1_m_s = (HY1b1_p4_s + HY1b2_p4_s).mass

HY2b1_p4 = TLorentzVectorArray.from_ptetaphim(HY2['b1']['pt'], HY2['b1']['eta'], HY2['b1']['phi'], HY2['b1']['m'])
HY2b2_p4 = TLorentzVectorArray.from_ptetaphim(HY2['b2']['pt'], HY2['b2']['eta'], HY2['b2']['phi'], HY2['b2']['m'])
HY2b1_p4_s = TLorentzVectorArray.from_ptetaphim(HY2_b1_pt_smeared, HY2['b1']['eta'], HY2['b1']['phi'], HY2['b1']['m'])
HY2b2_p4_s = TLorentzVectorArray.from_ptetaphim(HY2_b2_pt_smeared, HY2['b2']['eta'], HY2['b2']['phi'], HY2['b2']['m'])
HY2_m = (HY2b1_p4 + HY2b2_p4).mass
HY2_m_s = (HY2b1_p4_s + HY2b2_p4_s).mass

info("Preparing to plot.")
fig, axs = plt.subplots(nrows=3, ncols=1)

bins = np.linspace(0,200,100)

ax = axs[0]
ax.set_title(r'$H$')
# ax.hist(HX_m,  bins=bins, histtype='step', align='mid', label=r'$H_X$')
ax.hist(HX_m_s,  bins=bins, histtype='step', align='mid', label=r'smeared $H_X$')
ax.set_xlabel(r'$m_{bb}$ [GeV]')

ax = axs[1]
ax.set_title(r'$H_1$')
# ax.hist(HY1_m, bins=bins, histtype='step', align='mid', label=r'$H_{Y,1}$')
ax.hist(HY1_m_s,  bins=bins, histtype='step', align='mid', label=r'smeared $H_{Y,1}$')
ax.set_xlabel(r'$m_{bb}$ [GeV]')

ax = axs[2]
ax.set_title(r'$H_2$')
# ax.hist(HY2_m, bins=bins, histtype='step', align='mid', label=r'$H_{Y,2}$')
ax.hist(HY2_m_s,  bins=bins, histtype='step', align='mid', label=r'smeared $H_{Y,2}$')
ax.set_xlabel(r'$m_{bb}$ [GeV]')

plt.tight_layout()
plt.show()
fig.savefig("smeared_Higgs_mass.pdf", bbox_inches='tight')

b1X_m  = t['gen_HX_b1_m']
b2X_m  = t['gen_HX_b2_m']
b1Y1_m = t['gen_HY1_b1_m']
b2Y1_m = t['gen_HY1_b2_m']
b1Y2_m = t['gen_HY2_b1_m']
b2Y2_m = t['gen_HY2_b2_m']

# np.savez('6b_masses.npz', b1X_m=b1X_m, b2X_m=b2X_m, b1Y1_m=b1Y1_m, b2Y1_m=b2Y1_m, b1Y2_m=b1Y2_m, b2Y2_m=b2Y2_m)