from utils.analysis.signal import SixB
from utils.plotter import Hist
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

plot_dir = "plots/systematics/btag_reshaping_sfs"

maxbtag = SixB('/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM/btagsf/NMSSM_XYH_YToHH_6b_MX_700_MY_400_2M/ntuple.root', gnn_model='/eos/uscms/store/user/srosenzw/weaver/models/exp_sixb_official/feynnet_ranker_6b/20230731_7d266883bbfb88fe4e226783a7d1c9db_ranger_lr0.0047_batch2000_withbkg/predict_output/sf_NMSSM_XYH_YToHH_6b_MX_700_MY_400_2M.root')

bsf_central = maxbtag.get('bSFshape_central', library='np')
bins = np.arange(ak.min(maxbtag.n_jet) - 0.5, ak.max(maxbtag.n_jet) + 1.5)
scale = np.repeat(maxbtag.scale, len(maxbtag.n_jet))

n_before, e_b = np.histogram(maxbtag.n_jet.to_numpy(), bins=bins, weights=scale)
n_after, e_a  = np.histogram(maxbtag.n_jet.to_numpy(), bins=bins, weights=bsf_central*scale)
assert np.array_equal(e_a, e_b)

total_before = round(len(maxbtag.n_jet) * maxbtag.scale)
total_after  = round(bsf_central.sum() * maxbtag.scale)

x = (e_a[1:] + e_a[:-1]) / 2

fig, ax = plt.subplots()

n_b = Hist(x, weights=n_before, bins=e_a, ax=ax, label=f'Before SF ({total_before} Events)')
n_a = Hist(x, weights=n_after, bins=e_a, ax=ax, label=f'After SF ({total_after} Events)')
ax.set_xlabel('Jet Multiplicity')
ax.set_ylabel('AU')
fig.savefig(f'{plot_dir}/jet_mult.pdf')

ratio = n_b / n_a
ratio_dict = {int(e):n for e,n in zip(e_b[1:], ratio)}
print(ratio_dict)
# v_ratio = np.vectorize(get_ratio)
# final_bsf = v_ratio(maxbtag.n_jet.to_numpy()) * bsf_central

# print(final_bsf)