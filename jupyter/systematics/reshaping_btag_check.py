from utils.analysis.signal import SixB
from utils.plotter import Hist

import matplotlib.pyplot as plt
import numpy as np
import awkward as ak

import sys

plot_dir = "plots/systematics/btag_reshaping_sfs"

def get_ratio(val):
    return ratio_dict[val]

# tree = SixB('/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/btagsf/NMSSM_XToYHTo6B_MX-700_MY-400_TuneCP5_13TeV-madgraph-pythia8/ntuple.root', gnn_model='/eos/uscms/store/user/srosenzw/weaver/models/exp_sixb_official/feynnet_ranker_6b/20230731_7d266883bbfb88fe4e226783a7d1c9db_ranger_lr0.0047_batch2000_withbkg/predict_output/sf_NMSSM_XToYHTo6B_MX-700_MY-400_TuneCP5_13TeV-madgraph-pythia8.root')

# print(f".. found {round(len(tree.n_jet) * tree.scale)} events in sample")

bsf_central = tree.get('bSFshape_central', library='np')

fig, ax = plt.subplots()

n = Hist(bsf_central, bins=np.linspace(0.3,7,100), ax=ax, weights=tree.scale, label=f'Before ratio ({round(bsf_central.sum()*tree.scale)} Events)')
ax.set_xlabel('Central bSF')
ax.set_ylabel('AU')
fig.savefig(f'{plot_dir}/central_bsf_before.pdf')

bins = np.arange(5.99,17.99)
scale = np.repeat(tree.scale, len(tree.n_jet))

########################################################################
# Calculate SF corrective ratio in the inclusive region
########################################################################

print("---------- INCLUSIVE REGION CALCULATION ----------\n")

n_before, e_b = np.histogram(tree.n_jet.to_numpy(), bins=bins, weights=scale)
n_after, e_a  = np.histogram(tree.n_jet.to_numpy(), bins=bins, weights=bsf_central*scale)
assert np.array_equal(e_a, e_b)

total_before = round(len(tree.n_jet) * tree.scale)
total_after  = round(bsf_central.sum() * tree.scale)

x = (e_a[1:] + e_a[:-1]) / 2

fig, ax = plt.subplots()

n_b = Hist(x, weights=n_before, bins=e_a, ax=ax, label=f'Before SF ({total_before} Events)')
n_a = Hist(x, weights=n_after, bins=e_a, ax=ax, label=f'After SF ({total_after} Events)')
ax.set_xlabel('Jet Multiplicity')
ax.set_ylabel('AU')
fig.savefig(f'{plot_dir}/jet_mult.pdf')

ratio = n_b / n_a
ratio_dict = {int(e):n for e,n in zip(e_b[1:], ratio)}
# print(ratio_dict)
v_ratio = np.vectorize(get_ratio)
final_bsf = v_ratio(tree.n_jet.to_numpy()) * bsf_central

print(final_bsf)

fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})

ax = axs[0]
n_b = Hist(x, weights=n_before, bins=e_a, ax=ax, label=f'Before SF ({total_before} Events)')
n_a = Hist(x, weights=n_after, bins=e_a, ax=ax, label=f'After SF ({total_after} Events)')
ax.set_ylabel('AU')

ax = axs[1]
ax.plot(x, np.ones_like(x), '--', color='gray')
ax.scatter(x, ratio, color='k')
ax.set_xlabel('Jet Multiplicity')
ax.set_ylabel('Ratio\n[Before/After]', ha='center')
fig.savefig(f'{plot_dir}/jet_mult_ratio.pdf')

total_bsf_before = round((bsf_central).sum()*tree.scale)
total_bsf_after = round((final_bsf).sum()*tree.scale)

fig, ax = plt.subplots()

n = Hist(bsf_central, bins=np.linspace(0.3,7,100), ax=ax, weights=tree.scale, label=f'Before ratio ({total_bsf_before} Events)')
n = Hist(bsf_central, bins=np.linspace(0.3,7,100), ax=ax, weights=final_bsf*tree.scale, label=f'After ratio ({total_bsf_after} Events)')

ax.set_xlabel('Central bSF')
ax.set_ylabel('AU')
fig.savefig(f'{plot_dir}/central_bsf.pdf')

jet_btag = tree.jet_btag[:,:6]
final_bsf_jet, _ = ak.broadcast_arrays(ak.from_numpy(final_bsf), jet_btag)

fig, ax = plt.subplots()

n_b = Hist(ak.flatten(jet_btag), bins=np.linspace(0,1.01,101), ax=ax, label='Before SF', weights=tree.scale)
n_a = Hist(ak.flatten(jet_btag), bins=np.linspace(0,1.01,101), ax=ax, label='After SF & Correction', weights=ak.flatten(final_bsf_jet)*tree.scale)

# print(round(n_b.sum()), round(n_a.sum()))
# print(ak.min(jet_btag), ak.max(jet_btag))

ax.legend(loc=2)
ax.set_xlabel('b tag score')
ax.set_ylabel('AU')
fig.savefig(f'{plot_dir}/jet_btag.pdf')

tree.spherical_region()

jet_btag = tree.jet_btag[:,:6][tree.asr_mask]
final_bsf_jet, _ = ak.broadcast_arrays(ak.from_numpy(final_bsf[tree.asr_mask]), jet_btag)

fig, ax = plt.subplots()

n_b = Hist(ak.flatten(jet_btag), bins=np.linspace(0,1.01,101), ax=ax, label='Before SF', weights=tree.scale)
n_a = Hist(ak.flatten(jet_btag), bins=np.linspace(0,1.01,101), ax=ax, label='After SF & Correction', weights=ak.flatten(final_bsf_jet)*tree.scale)

# print(round(n_b.sum()), round(n_a.sum()))
# print(ak.min(jet_btag), ak.max(jet_btag))

ax.legend(loc=2)
ax.set_xlabel('b tag score in SR')
ax.set_ylabel('AU')
fig.savefig(f'{plot_dir}/jet_btag_asr.pdf')

########################################################################
# Calculate SF corrective ratio in the signal region
########################################################################

print("---------- SIGNAL REGION CALCULATION ----------\n")

scale = np.repeat(tree.scale, len(tree.n_jet[tree.asr_mask]))

n_before, e_b = np.histogram(tree.n_jet.to_numpy()[tree.asr_mask], bins=bins, weights=scale)
n_after, e_a  = np.histogram(tree.n_jet.to_numpy()[tree.asr_mask], bins=bins, weights=bsf_central[tree.asr_mask]*scale)
assert np.array_equal(e_a, e_b)

print(f"ratio 1 = {n_before/n_after}")

total_before = round(len(tree.n_jet[tree.asr_mask]) * tree.scale)
total_after  = round(bsf_central[tree.asr_mask].sum() * tree.scale)

x = (e_a[1:] + e_a[:-1]) / 2

fig, ax = plt.subplots()

n_b = Hist(x, weights=n_before, bins=e_a, ax=ax, label=f'Before SF ({total_before} Events)')
n_a = Hist(x, weights=n_after, bins=e_a, ax=ax, label=f'After SF ({total_after} Events)')
ax.set_xlabel('Jet Multiplicity')
ax.set_ylabel('AU')
fig.savefig(f'{plot_dir}/asr_ratio_jet_mult.pdf')

ratio = n_b / n_a
# print(f"ratio 2 = {ratio}")
ratio_dict = {int(e):n for e,n in zip(e_b[1:], ratio)}
# print(ratio_dict)
v_ratio = np.vectorize(get_ratio)
final_bsf = v_ratio(tree.n_jet.to_numpy()) * bsf_central

print(final_bsf)

fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})

ax = axs[0]
n_b = Hist(x, weights=n_before, bins=e_a, ax=ax, label=f'Before SF ({total_before} Events)')
n_a = Hist(x, weights=n_after, bins=e_a, ax=ax, label=f'After SF ({total_after} Events)')
ax.set_ylabel('AU')

ax = axs[1]
ax.plot(x, np.ones_like(x), '--', color='gray')
ax.scatter(x, ratio, color='k')
ax.set_xlabel('Jet Multiplicity')
ax.set_ylabel('Ratio\n[Before/After]', ha='center')
fig.savefig(f'{plot_dir}/asr_ratio_jet_mult_ratio.pdf')

total_bsf_before = round((bsf_central).sum()*tree.scale)
total_bsf_after = round((final_bsf).sum()*tree.scale)

fig, ax = plt.subplots()

n = Hist(bsf_central, bins=np.linspace(0.3,7,100), ax=ax, weights=tree.scale, label=f'Before ratio ({total_bsf_before} Events)')
n = Hist(bsf_central, bins=np.linspace(0.3,7,100), ax=ax, weights=final_bsf*tree.scale, label=f'After ratio ({total_bsf_after} Events)')

ax.set_xlabel('Central bSF')
ax.set_ylabel('AU')
fig.savefig(f'{plot_dir}/asr_ratio_central_bsf.pdf')

jet_btag = tree.jet_btag[tree.asr_mask][:,:6]
final_bsf_jet, _ = ak.broadcast_arrays(ak.from_numpy(final_bsf[tree.asr_mask]), jet_btag)

fig, ax = plt.subplots()

n_b = Hist(ak.flatten(jet_btag), bins=np.linspace(0,1.01,101), ax=ax, label='Before SF', weights=tree.scale)
n_a = Hist(ak.flatten(jet_btag), bins=np.linspace(0,1.01,101), ax=ax, label='After SF & Correction', weights=ak.flatten(final_bsf_jet)*tree.scale)

# print(round(n_b.sum()), round(n_a.sum()))
# print(ak.min(jet_btag), ak.max(jet_btag))

ax.legend(loc=2)
ax.set_xlabel('b tag score')
ax.set_ylabel('AU')
fig.savefig(f'{plot_dir}/asr_ratio_jet_btag.pdf')

tree.spherical_region()

jet_btag = tree.jet_btag[:,:6][tree.asr_mask]
final_bsf_jet, _ = ak.broadcast_arrays(ak.from_numpy(final_bsf[tree.asr_mask]), jet_btag)

fig, ax = plt.subplots()

n_b = Hist(ak.flatten(jet_btag), bins=np.linspace(0,1.01,101), ax=ax, label='Before SF', weights=tree.scale)
n_a = Hist(ak.flatten(jet_btag), bins=np.linspace(0,1.01,101), ax=ax, label='After SF & Correction', weights=ak.flatten(final_bsf_jet)*tree.scale)

# print(round(n_b.sum()), round(n_a.sum()))
# print(ak.min(jet_btag), ak.max(jet_btag))

ax.legend(loc=2)
ax.set_xlabel('b tag score in SR')
ax.set_ylabel('AU')
fig.savefig(f'{plot_dir}/asr_ratio_jet_btag_asr.pdf')