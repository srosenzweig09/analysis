
print("Loading packages...")
from utils.analysis import SixB
from utils.plotter import Hist2d
from utils.files import *
from utils.bashUtils import suppress_stdout

from configparser import ConfigParser
from matplotlib.backends.backend_pdf import PdfPages
from rich import print as rprint
from rich.console import Console
console = Console()

import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess, shlex
from tqdm import tqdm

info_dict = {}

savedir = f'plots/1_signal_exploration/resolved'

# load the signal
base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM'
cmd = f"ls {base}"
output = subprocess.check_output(shlex.split(cmd))
output = output.decode('UTF-8')
output = output.split('\n')
output = [f"{base}/{out}/ntuple.root" for out in output if out.startswith('NMSSM')]

# print(output)

signal = []
with console.status("[bold][green]Loading signal...") as status:
    # with suppress_stdout():
    signal = [SixB(out) for out in output]
    # for out in output:
        # try: signal.append(SixB(out))
        # except: print(f"[red]Failed to load: {out}")

def get_higgs_pt(sig):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(26,8))
    mx, my = sig.mx, sig.my

    nmax = []
    p = round(np.sum(sig.gen_matched_HX.pt[sig.good_mask] > 350)/np.sum(sig.gen_matched_HX.pt[sig.good_mask] > 0) * 100, 1)
    n = sig.hist(sig.gen_matched_HX.pt[sig.good_mask], bins=np.linspace(0,600,31), ax=axs[0], density=True, label=f"HX ({p}% above 350 GeV)")
    nmax.append(n.max())
    p = round(np.sum(sig.gen_matched_H1.pt[sig.good_mask] > 350)/np.sum(sig.gen_matched_HX.pt[sig.good_mask] > 0) * 100, 1)
    n = sig.hist(sig.gen_matched_H1.pt[sig.good_mask], bins=np.linspace(0,600,31), ax=axs[1], density=True, label=f"H1 ({p}% above 350 GeV)")
    nmax.append(n.max())
    p = round(np.sum(sig.gen_matched_H2.pt[sig.good_mask] > 350)/np.sum(sig.gen_matched_HX.pt[sig.good_mask] > 0) * 100, 1)
    n = sig.hist(sig.gen_matched_H2.pt[sig.good_mask], bins=np.linspace(0,600,31), ax=axs[2], density=True, label=f"H2 ({p}% above 350 GeV)")
    nmax.append(n.max())

    for ax,n in zip(axs,nmax):
        ax.set_xlabel(r"$p_T$ [GeV]")
        ax.plot([350,350],[0,n],color='gray',linestyle='--')
        ax.legend()
    return fig, axs

def get_higgs_dr(sig):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(26,8))
    mx, my = sig.mx, sig.my

    nmax = []
    p = round(np.sum(sig.gen_matched_HX.dr[sig.good_mask] < 0.4)/np.sum(sig.gen_matched_HX.pt[sig.good_mask] > 0) * 100, 1)
    n = sig.hist(sig.gen_matched_HX.dr[sig.good_mask], bins=np.linspace(0,4,31), ax=axs[0], density=True, label=f"HX ({p}% below 0.4)")
    nmax.append(n.max())
    p = round(np.sum(sig.gen_matched_H1.dr[sig.good_mask] < 0.4)/np.sum(sig.gen_matched_HX.pt[sig.good_mask] > 0) * 100, 1)
    n = sig.hist(sig.gen_matched_H1.dr[sig.good_mask], bins=np.linspace(0,4,31), ax=axs[1], density=True, label=f"HX ({p}% below 0.4)")
    nmax.append(n.max())
    p = round(np.sum(sig.gen_matched_H2.dr[sig.good_mask] < 0.4)/np.sum(sig.gen_matched_HX.pt[sig.good_mask] > 0) * 100, 1)
    n = sig.hist(sig.gen_matched_H2.dr[sig.good_mask], bins=np.linspace(0,4,31), ax=axs[2], density=True, label=f"HX ({p}% below 0.4)")
    nmax.append(n.max())

    for ax,n in zip(axs,nmax):
        ax.set_xlabel(r"$\Delta R$")
        ax.plot([0.4,0.4],[0,n],color='gray',linestyle='--')
        ax.legend()
    return fig, axs

def get_higgs_2d_pt_dr(sig):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(26,8))
    mx, my = sig.mx, sig.my

    n, ex, ey, im1 = Hist2d(sig.gen_matched_HX.pt[sig.good_mask], sig.gen_matched_HX.dr[sig.good_mask], bins=(np.linspace(0,600,31), np.linspace(0,4,31)), ax=axs[0], density=True, label=f"[{mx},{my}]")
    n, ex, ey, im2 = Hist2d(sig.gen_matched_H1.pt[sig.good_mask], sig.gen_matched_H1.dr[sig.good_mask], bins=(np.linspace(0,600,31), np.linspace(0,4,31)), ax=axs[1], density=True, label=f"[{mx},{my}]")
    n, ex, ey, im3 = Hist2d(sig.gen_matched_H2.pt[sig.good_mask], sig.gen_matched_H2.dr[sig.good_mask], bins=(np.linspace(0,600,31), np.linspace(0,4,31)), ax=axs[2], density=True, label=f"[{mx},{my}]")

    for ax,im in zip(axs,[im1,im2,im3]):
        ax.set_xlabel(r"$p_T$ [GeV]")
        ax.set_ylabel(r"$\Delta R$")
        ax.set_title(sig.sample)
        fig.colorbar(im, ax=ax)
        ax.plot([0,600],[0.4,0.4],color='gray',linestyle='--',lw=3)
        ax.plot([350,350],[0,4],color='gray',linestyle='--',lw=3)
    return fig, axs

print("..generating resolved mh, my plots")
with PdfPages(f"{savedir}/resolved_studies_pt_dR.pdf") as pdf:
    # for sig in tqdm(signal):
    for out in tqdm(output):
        sig = SixB(out)
        rprint(f"Processing: mx = {sig.mx}, my = {sig.my}")
        sig.initialize_gen()
        fig, axs = get_higgs_pt(sig)
        pdf.savefig(dpi=300)
        plt.close()
        fig, axs = get_higgs_dr(sig)
        pdf.savefig(dpi=300)
        plt.close()
        fig, axs = get_higgs_2d_pt_dr(sig)
        pdf.savefig(dpi=300)
        plt.close()
print("DONE")