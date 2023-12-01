import numpy as np
import matplotlib.pyplot as plt
from utils.analysis import sixb_from_gnn, Bkg
from utils.files import get_NMSSM_list, get_qcd_ttbar
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import awkward as ak
from rich import print as rprint
from rich.console import Console
console = Console()
import sys

# signal = [sixb_from_gnn(out) for out in get_NMSSM_list()]

bkg = Bkg(get_qcd_ttbar('maxbtag_4b'))
bkg.spherical_region(cfg='config/bdt_params.cfg')

with PdfPages('plots/6_background_modeling/btag_motivation.pdf') as pdf:
    # for sig in tqdm(signal):
    for out in tqdm(get_NMSSM_list()):
        console.log(f"Processing: {out}")
        
        sig = sixb_from_gnn(out)
        sig.spherical_region()

        fig, axs = plt.subplots(ncols=4, figsize=(40,10))

        # --------------------
        # average b tag score

        ax = axs[0]

        bins = np.linspace(0,1.01,31)
        n_bkg = bkg.hist(bkg.btag_avg, mask=bkg.acr_mask, bins=bins, density=True, ax=ax)
        n_sig = sig.hist(sig.btag_avg[sig.asr_mask], bins=bins, density=True, ax=ax)

        mask = n_bkg > n_sig
        cut = bins[1:][mask][-1]

        bkg_rej = ak.sum(bkg.btag_avg[bkg.acr_mask] < cut)/bkg.acr_mask.sum()
        sig_ret = ak.sum(sig.btag_avg[sig.asr_mask] >= cut)/sig.asr_mask.sum()

        handle1 = Line2D([0], [0], color='k', lw=2, label=f"bkg (rej: {int(bkg_rej*100)}%)")
        handle2 = Line2D([0], [0], color='C0', lw=2, label=f"sig (keep: {int(sig_ret*100)}%)")
        ax.legend(handles=[handle1, handle2])

        ax.set_xlabel('average b tag score of six jets')

        ax.plot([cut, cut],[0, max(n_bkg.max(), n_sig.max())], linestyle='--', color='grey')

        # --------------------
        # n tight

        ax = axs[1]

        bins = np.arange(10)
        n_bkg = bkg.hist(bkg.n_tight, mask=bkg.acr_mask, bins=bins, density=True, ax=ax)
        n_sig = sig.hist(sig.n_tight[sig.asr_mask], bins=bins, density=True, ax=ax)

        mask = n_bkg > n_sig
        cut = bins[1:][mask][-1]

        bkg_rej = ak.sum(bkg.n_tight[bkg.acr_mask] < cut)/bkg.acr_mask.sum()
        sig_ret = ak.sum(sig.n_tight[sig.asr_mask] >= cut)/sig.asr_mask.sum()

        handle1 = Line2D([0], [0], color='k', lw=2, label=f"bkg (rej: {int(bkg_rej*100)}%)")
        handle2 = Line2D([0], [0], color='C0', lw=2, label=f"sig (keep: {int(sig_ret*100)}%)")
        ax.legend(handles=[handle1, handle2])

        ax.set_xlabel('number of tight b-tagged jets in six jet selection')

        ax.plot([cut, cut],[0, max(n_bkg.max(), n_sig.max())], linestyle='--', color='grey')

        # --------------------
        # n medium

        ax = axs[2]

        bins = np.arange(10)
        bkg_total = bkg.n_tight + bkg.n_medium
        sig_total = sig.n_tight + sig.n_medium
        n_bkg = bkg.hist(bkg_total, mask=bkg.acr_mask, bins=bins, density=True, ax=ax)
        n_sig = sig.hist(sig_total[sig.asr_mask], bins=bins, density=True, ax=ax)

        mask = n_bkg > n_sig
        cut = bins[1:][mask][-1]

        bkg_rej = ak.sum(bkg_total[bkg.acr_mask] < cut)/bkg.acr_mask.sum()
        sig_ret = ak.sum(sig_total[sig.asr_mask] >= cut)/sig.asr_mask.sum()

        handle1 = Line2D([0], [0], color='k', lw=2, label=f"bkg (rej: {int(bkg_rej*100)}%)")
        handle2 = Line2D([0], [0], color='C0', lw=2, label=f"sig (keep: {int(sig_ret*100)}%)")
        ax.legend(handles=[handle1, handle2])

        ax.set_xlabel('number of medium b-tagged jets in six jet selection')

        ax.plot([cut, cut],[0, max(n_bkg.max(), n_sig.max())], linestyle='--', color='grey')

        # --------------------
        # n loose
        ax = axs[3]

        bins = np.arange(10)
        bkg_total = bkg.n_tight + bkg.n_medium + bkg.n_loose
        sig_total = sig.n_tight + sig.n_medium + sig.n_loose
        n_bkg = bkg.hist(bkg_total, mask=bkg.acr_mask, bins=bins, density=True, ax=ax)
        n_sig = sig.hist(sig_total[sig.asr_mask], bins=bins, density=True, ax=ax)

        mask = n_bkg > n_sig
        cut = bins[1:][mask][-1]

        bkg_rej = ak.sum(bkg_total[bkg.acr_mask] < cut)/bkg.acr_mask.sum()
        sig_ret = ak.sum(sig_total[sig.asr_mask] >= cut)/sig.asr_mask.sum()

        handle1 = Line2D([0], [0], color='k', lw=2, label=f"bkg (rej: {int(bkg_rej*100)}%)")
        handle2 = Line2D([0], [0], color='C0', lw=2, label=f"sig (keep: {int(sig_ret*100)}%)")
        ax.legend(handles=[handle1, handle2])

        ax.set_xlabel('number of loose b-tagged jets in six jet selection')

        ax.plot([cut, cut],[0, max(n_bkg.max(), n_sig.max())], linestyle='--', color='grey')

        pdf.savefig()
        plt.close()
