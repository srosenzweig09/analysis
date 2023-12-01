from utils.analysis.signal import SixB
from utils.plotter import Hist

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import awkward as ak

import sys

plot_dir = "plots/systematics/btag_reshaping_sfs"

tree = SixB('/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/btagsf/NMSSM_XToYHTo6B_MX-700_MY-400_TuneCP5_13TeV-madgraph-pythia8/ntuple.root', gnn_model='/eos/uscms/store/user/srosenzw/weaver/models/exp_sixb_official/feynnet_ranker_6b/20230731_7d266883bbfb88fe4e226783a7d1c9db_ranger_lr0.0047_batch2000_withbkg/predict_output/2018/sf_NMSSM_XToYHTo6B_MX-700_MY-400_TuneCP5_13TeV-madgraph-pythia8.root')
# tree = SixB('/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM/btagsf/NMSSM_XYH_YToHH_6b_MX_700_MY_400_2M/ntuple.root', gnn_model='/eos/uscms/store/user/srosenzw/weaver/models/exp_sixb_official/feynnet_ranker_6b/20230731_7d266883bbfb88fe4e226783a7d1c9db_ranger_lr0.0047_batch2000_withbkg/predict_output/sf_NMSSM_XToYHTo6B_MX-700_MY-400_TuneCP5_13TeV-madgraph-pythia8.root')

bsf_central = tree.get('bSFshape_central', library='np')
print(bsf_central)

ratio_dict = {6: 1.0523819889911639, 7: 1.0407451011402493, 8: 1.0275378560570592, 9: 1.0085806355366151, 10: 0.9882910129816392, 11: 0.9655276469570114, 12: 0.9491447971301967, 13: 0.93976147195616, 14: 0.8796787950777801, 15: 0.9415719956127135, 16: 0.9129294828487661, 17: 1.0510190939357809, 18: 0.4943583701844633}

get_ratio = lambda x : ratio_dict[x]
get_ratio = np.vectorize(get_ratio)
final_bsf = final_bsf = v_ratio(tree.n_jet.to_numpy()) * bsf_central

n_before, e = np.histogram(tree.n_jet.to_numpy(), bins=bins, weights=scale)
n_after, e  = np.histogram(tree.n_jet.to_numpy(), bins=bins, weights=bsf_central*scale)

n_b = Hist(x, weights=n_before, bins=e, ax=ax, label=f'Before SF ({total_before} Events)')
n_a = Hist(x, weights=n_after, bins=e, ax=ax, label=f'After SF ({total_after} Events)')
ax.set_xlabel('Jet Multiplicity')
ax.set_ylabel('AU')
fig.savefig(f'{plot_dir}/jet_mult.pdf')



sys.exit()




# print(tree.asr_bSFshape_central)

bins = np.arange(16)

fig, ax = plt.subplots()

systematics_name = [key for key in tree.keys() if key.startswith('bSFshape_')]
mask_names = ['asr', 'acr', 'vsr', 'vcr']
masks = [tree.asr_mask, tree.acr_mask, tree.vsr_mask, tree.vcr_mask]

for mask,region in zip(masks,mask_names):
    for sys_name in systematics_name:
        branch_name = f"{region}_{sys_name}"
        sf = getattr(tree, branch_name)

        fig, ax = plt.subplots()

        n_med = ak.sum(tree.medium_mask[mask], axis=1).to_numpy()
        n_med_bef = Hist(n_med, bins=np.arange(10), ax=ax, weights=tree.scale, label=f'Before SFs')
        n_med_aft = Hist(n_med, bins=np.arange(10), ax=ax, weights=sf*tree.scale, label=f'After SFs + correction')
        ax.set_xlabel('Medium Jet Multiplicity')
        ax.set_ylabel('AU')
        handle1 = Line2D([0], [0], color='C0', lw=2, label=f'Before SFs ({round(n_med_bef.sum())} jets)')
        handle2 = Line2D([0], [0], color='C1', lw=2, label=f'After SFs+corr ({round(n_med_aft.sum())} jets)')
        ax.legend(handles=[handle1, handle2])
        if 'asr' in branch_name and 'central' in branch_name: fig.savefig('plots/systematics/btag_reshaping_sfs/medium_jets.pdf')
        plt.close()
        
        fig, ax = plt.subplots()
        n_tight = ak.sum(tree.tight_mask[mask], axis=1).to_numpy()
        n_tight_bef = Hist(n_tight, bins=np.arange(10), ax=ax, weights=tree.scale, label=f'Before SFs')
        n_tight_aft = Hist(n_tight, bins=np.arange(10), ax=ax, weights=sf*tree.scale, label=f'After SFs + correction')
        ax.set_xlabel('Tight Jet Multiplicity')
        ax.set_ylabel('AU')
        handle1 = Line2D([0], [0], color='C0', lw=2, label=f'Before SFs ({round(n_tight_bef.sum())} jets)')
        handle2 = Line2D([0], [0], color='C1', lw=2, label=f'After SFs+corr ({round(n_tight_aft.sum())} jets)')
        ax.legend(handles=[handle1, handle2])
        if 'asr' in branch_name and 'central' in branch_name: fig.savefig('plots/systematics/btag_reshaping_sfs/tight_jets.pdf')
        plt.close()

        # print(branch_name)
        # print(f"{np.around(n_med_bef)} =? {np.around(n_med_aft)}")
        # print(f"{np.around(n_tight_bef)} =? {np.around(n_tight_aft)}")


        # n = [ak.sum(tree.fail_mask[mask], axis=1), ak.sum(tree.loose_mask[mask]), ak.sum(tree.medium_mask[mask]), ak.sum(tree.tight_mask[mask])]
        # n = np.array([round(x*tree.scale) for x in n])

        # tmp_n = [ak.sum(tree.fail_mask[mask]*sf), ak.sum(tree.loose_mask[mask]*sf), ak.sum(tree.medium_mask[mask]*sf), ak.sum(tree.tight_mask[mask]*sf)]
        # tmp_n = np.array([round(x*tree.scale) for x in n])
        # print(f"{n} =? {tmp_n}")