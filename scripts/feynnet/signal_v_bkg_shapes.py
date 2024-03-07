# python scripts/feynnet/signal_v_bkg_shapes.py

from utils import *
import utils.analysis.signal as uas
import matplotlib.pyplot as plt
import os, sys, shutil
from matplotlib.backends.backend_pdf import PdfPages
import random

rand_num = random.randint(0,1000000)
cfg = f'tmp/{rand_num}_feynnet.cfg'
shutil.copyfile('config/feynnet.cfg', cfg)

mpoints = [
    [450,300],
    [700,500],
    [1000,300],
    [1000,600],
    [1200,300],
    [1200,1000]
]

density = True

bkg = uas.Bkg(feyn=cfg)
signals = [SixB(get_NMSSM(*m), feyn=cfg) for m in mpoints]
version = signals[0].model.version

savepath = signals[0].model.savepath
if not os.path.exists(savepath): os.makedirs(savepath)
print(savepath)

qcd_color = 'wheat'
ttbar_color = 'lightblue'

if version == 'new':
    with PdfPages(f'{savepath}/sorted_rank.pdf') as pdf:
        for i,signal in enumerate(signals):
            fig, ax = plt.subplots(figsize=(8,6))
            bins = np.linspace(-0.5,1.5,61)

            n = Hist(bkg.qcd.rank, bins=bins, ax=ax, density=density, label='qcd', histtype='stepfilled', color=qcd_color, weights=bkg.qcd.nomWeight, scale=bkg.ratio_qcd)

            n = Hist(bkg.ttbar.rank, bins=bins, ax=ax, density=density, bottom=n, label='ttbar', histtype='stepfilled', color=ttbar_color, weights=bkg.ttbar.nomWeight, scale=bkg.ratio_ttbar)

            n = Hist(signal.rank, bins=bins, ax=ax, density=density, label=signal.sample, weights=signal.nomWeight)
            
            ax.set_title(signal.sample)
            ax.set_xlabel('Leading Rank Score')
            ax.set_ylabel('AU')
            ax.legend(loc=2)

            fig.suptitle(bkg.ttbar.model_name, y=1.0)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()


xlabels = [f'Higgs {i+1} Mass [GeV]' for i in range(3)]
with PdfPages(f'{savepath}/reco_h_masses.pdf') as pdf:
    bins = np.linspace(0,300,41)

    for i,signal in enumerate(signals):
        signal.model.init_particles(signal, combo=14)

        fig, axs = plt.subplots(ncols=3, figsize=(30,8))
        ax = axs[0]
        n = Hist(bkg.qcd.H1_m, bins=bins, ax=ax, weights=bkg.qcd.nomWeight, label='qcd', density=density, histtype='stepfilled', color=qcd_color, scale=bkg.ratio_qcd)
        n = Hist(bkg.ttbar.H1.m, bins=bins, ax=ax, weights=bkg.ttbar.nomWeight, bottom=n, label='ttbar', density=density, histtype='stepfilled', color=ttbar_color, scale=bkg.ratio_ttbar)
        n = Hist(signal.H1.m, bins=bins, ax=ax, weights=signal.nomWeight, density=density, label=signal.sample)
        nmax = n.max()
        # n = Hist(signal.H1.m, bins=bins, ax=ax, weights=signal.nomWeight, density=density, label='first rank')
        # n = Hist(signal.model.H1.m, bins=bins, ax=ax, weights=signal.nomWeight, density=density, label='last rank')
        
        ax = axs[1]
        n = Hist(bkg.qcd.H2_m, bins=bins, ax=ax, weights=bkg.qcd.nomWeight, density=density, label='qcd', histtype='stepfilled', color=qcd_color, scale=bkg.ratio_qcd)
        n = Hist(bkg.ttbar.H2.m, bins=bins, ax=ax, weights=bkg.ttbar.nomWeight, density=density, bottom=n, label='ttbar', histtype='stepfilled', color=ttbar_color, scale=bkg.ratio_ttbar)
        n = Hist(signal.H2.m, bins=bins, ax=ax, weights=signal.nomWeight, density=density, label=signal.sample)
        nmax = max(nmax, n.max())
        # n = Hist(signal.H2.m, bins=bins, ax=ax, weights=signal.nomWeight, density=density, label='first rank')
        # n = Hist(signal.model.H2.m, bins=bins, ax=ax, weights=signal.nomWeight, density=density, label='last rank')
        
        ax = axs[2]
        n = Hist(bkg.qcd.H3_m, bins=bins, ax=ax, weights=bkg.qcd.nomWeight, density=density, label='qcd', histtype='stepfilled', color=qcd_color, scale=bkg.ratio_qcd)
        n = Hist(bkg.ttbar.H3.m, bins=bins, ax=ax, weights=bkg.ttbar.nomWeight, density=density, bottom=n, label='ttbar', histtype='stepfilled', color=ttbar_color, scale=bkg.ratio_ttbar)
        n = Hist(signal.H3.m, bins=bins, ax=ax, weights=signal.nomWeight, density=density, label=signal.sample)
        nmax = max(nmax, n.max())
        # n = Hist(signal.H3.m, bins=bins, ax=ax, weights=signal.nomWeight, density=density, label='first rank')
        # n = Hist(signal.model.H3.m, bins=bins, ax=ax, weights=signal.nomWeight, density=density, label='last rank')

        for j,ax in enumerate(axs):
            ax.set_ylim(0, nmax*1.1)
            ax.set_title(signal.sample)
            ax.set_xlabel(xlabels[j])
            ax.legend()

        fig.suptitle(signal.model_name, y=1.0)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


with PdfPages(f'{savepath}/reco_h_masses_2d.pdf') as pdf:
    for i,tree in enumerate([bkg.qcd, bkg.ttbar] + signals):
        fig, ax = plt.subplots(figsize=(10,8))
        bins = np.linspace(0,300,41)

        try: hx,h1,h2 = tree.H1.m, tree.H2.m, tree.H3.m
        except: hx,h1,h2 = tree.H1_m, tree.H2_m, tree.H3_m
        if not isinstance(hx, np.ndarray): hx = hx.to_numpy()
        if not isinstance(h1, np.ndarray): h1 = h1.to_numpy()
        if not isinstance(h2, np.ndarray): h2 = h2.to_numpy()
        weight = tree.nomWeight.to_numpy()

        n, im = Hist2d(hx, h1, bins=bins, ax=ax, weights=weight)

        ax.set_title(tree.sample)
        fig.colorbar(im, ax=ax)

        if version == 'new':
            fig.suptitle(bkg.ttbar.model_name, y=1.0)
        elif version == 'old':
            fig.suptitle(bkg.ttbar.model_name, y=1.0, fontsize=12)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

os.remove(cfg)

# with PdfPages(f'{savepath}/avg_btag.pdf') as pdf:
#     for i,signal in enumerate(signals):
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