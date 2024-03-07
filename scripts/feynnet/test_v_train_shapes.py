from utils import *
import utils.analysis.signal as uas
import matplotlib.pyplot as plt
import os, sys
from matplotlib.backends.backend_pdf import PdfPages

mpoints = [
    [450,300],
    [700,500],
    [1000,300],
    [1000,600],
    [1200,300],
    [1200,1000]
]

density = True

test_sig = [SixB(get_NMSSM(*m)) for m in mpoints]
train_sig = [SixB(get_NMSSM(*m, selection='maxbtag', private=True)) for m in mpoints]

version = test_sig[0].model.version
model_name = test_sig[0].model.model_name
savepath = test_sig[0].model.savepath
if not os.path.exists(savepath): os.makedirs(savepath)
print(savepath)

qcd_color = 'wheat'
ttbar_color = 'lightblue'

xlabels = [f'Higgs {i+1} Mass [GeV]' for i in range(3)]
with PdfPages(f'{savepath}/train_reco_h_masses.pdf') as pdf:
    bins = np.linspace(0,300,41)

    for test,train in zip(test_sig, train_sig):
        fig, axs = plt.subplots(ncols=3, figsize=(30,8))
        ax = axs[0]
        n = Hist(test.H1.m, bins=bins, ax=ax, weights=test.nomWeight, density=density, label=test.sample)
        n = Hist(train.H1.m, bins=bins, ax=ax, weights=train.nomWeight, density=density, label=train.sample, histtype='stepfilled', color='pink')
        
        ax = axs[1]
        n = Hist(test.H2.m, bins=bins, ax=ax, weights=test.nomWeight, density=density, label=test.sample)
        n = Hist(train.H2.m, bins=bins, ax=ax, weights=train.nomWeight, density=density, label=train.sample, histtype='stepfilled', color='pink')
        
        ax = axs[2]
        n = Hist(test.H3.m, bins=bins, ax=ax, weights=test.nomWeight, density=density, label=test.sample)
        n = Hist(train.H3.m, bins=bins, ax=ax, weights=train.nomWeight, density=density, label=train.sample, histtype='stepfilled', color='pink')

        for j,ax in enumerate(axs):
            ax.set_title(test.sample)
            ax.set_xlabel(xlabels[j])
            ax.legend()

        fig.suptitle(model_name, y=1.0)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()



# with PdfPages(f'{savepath}/reco_h_masses_2d.pdf') as pdf:
#     for i,tree in enumerate([bkg.qcd, bkg.ttbar] + test_sig):
#         fig, ax = plt.subplots(figsize=(10,8))
#         bins = np.linspace(0,300,41)

#         try: hx,h1,h2 = tree.H1.m, tree.H2.m, tree.H3.m
#         except: hx,h1,h2 = tree.H1_m, tree.H2_m, tree.H3_m
#         if not isinstance(hx, np.ndarray): hx = hx.to_numpy()
#         if not isinstance(h1, np.ndarray): h1 = h1.to_numpy()
#         if not isinstance(h2, np.ndarray): h2 = h2.to_numpy()
#         weight = tree.nomWeight.to_numpy()

#         n, im = Hist2d(hx, h1, bins=bins, ax=ax, weights=weight)

#         ax.set_title(tree.sample)
#         fig.colorbar(im, ax=ax)

#         if version == 'new':
#             fig.suptitle(bkg.ttbar.model_name, y=1.0)
#         elif version == 'old':
#             fig.suptitle(bkg.ttbar.model_name, y=1.0, fontsize=12)

#         plt.tight_layout()
#         pdf.savefig(fig, bbox_inches='tight')
#         plt.close()
#         # fig.savefig(f'{savepath}/reco_higgs_masses_2d.pdf', bbox_inches='tight')



# with PdfPages(f'{savepath}/avg_btag.pdf') as pdf:
#     for i,signal in enumerate(test_sig):
#         fig, ax = plt.subplots(figsize=(8,6))
#         bins = np.linspace(0,1,61)

#         n = Hist(bkg.qcd.btag_avg[bkg.qcd.asr_mask], bins=bins, ax=ax, density=density, label='qcd', histtype='stepfilled', color=qcd_color, weights=bkg.qcd.nomWeight[bkg.qcd.asr_mask], scale=bkg.ratio_qcd)

#         n = Hist(bkg.ttbar.btag_avg[bkg.ttbar.asr_mask], bins=bins, ax=ax, density=density, bottom=n, label='ttbar', histtype='stepfilled', color=ttbar_color, weights=bkg.ttbar.nomWeight[bkg.ttbar.asr_mask], scale=bkg.ratio_ttbar)

#         n = Hist(signal.btag_avg[signal.asr_mask], bins=bins, ax=ax, density=density, label=signal.sample, weights=signal.nomWeight[signal.asr_mask])
        
#         ax.set_title(signal.sample)
#         ax.set_xlabel('Average b tag score')
#         ax.set_ylabel('AU')
#         ax.legend(loc=2)

#         if version == 'new':
#             fig.suptitle(bkg.ttbar.model_name, y=1.0)
#         elif version == 'old':
#             fig.suptitle(bkg.ttbar.model_name, y=1.0, fontsize=12)

#         plt.tight_layout()
#         pdf.savefig(fig, bbox_inches='tight')
#         plt.close()
#         # fig.savefig(f'{savepath}/avg_btag.pdf', bbox_inches='tight')