from matplotlib.backends.backend_pdf import PdfPages
from utils.analysis import SixB, Bkg
import matplotlib.pyplot as plt
from rich import print as rprint
from rich.console import Console
console = Console()
from tqdm import tqdm
# from utils.plotter import Hist
import numpy as np
from utils.filelists import *
import os, sys

from utils.analysis.gnn import model_path
if model_path[-1] == '/': model_path = model_path[:-1]
model_dir = model_path.split('/')[-2]
model_savein = f'plots/feynnet/{model_dir}'
print(model_dir)

year = 'Summer2018UL'
yr = int(year.split('Summer')[1].split('UL')[0])

model_savein = f'plots/feynnet/{model_dir}'
if not os.path.exists(model_savein): os.makedirs(model_savein)
model_savein = f'plots/feynnet/{model_dir}/{year}'
if not os.path.exists(model_savein): os.makedirs(model_savein)

info_dict = {}

# # load the signal
# base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM'
# cmd = f"ls {base}"
# output = subprocess.check_output(shlex.split(cmd))
# output = output.decode('UTF-8')
# output = output.split('\n')
# output = [f"{base}/{out}/ntuple.root" for out in output if out.startswith('NMSSM') if int(out.split('-')[1].split('_')[0]) < 2100]

# import sys
# sys.exit()

bkg = Bkg(get_qcd_ttbar(selection='maxbtag_4b', run=year), year=yr)

nbins = 31
h_bins = np.linspace(0,300,nbins)
y_bins = np.linspace(0,2000,nbins)
x_bins = np.linspace(0,2200,nbins)

fig, axs = plt.subplots(ncols=5, figsize=(50,10))

ax = axs[0]
bkg.hist(bkg.HX.m, h_bins, ax=ax, label='QCD+ttbar', color='grey', density=True)
ax.set_xlabel(r'$H_X$ Mass [GeV]')

ax = axs[1]
bkg.hist(bkg.H1.m, h_bins, ax=ax, label='QCD+ttbar', color='grey', density=True)
ax.set_xlabel(r'$H_1$ Mass [GeV]')

ax = axs[2]
bkg.hist(bkg.H2.m, h_bins, ax=ax, label='QCD+ttbar', color='grey', density=True)
ax.set_xlabel(r'$H_2$ Mass [GeV]')

ax = axs[3]
bkg.hist(bkg.Y.m, y_bins, ax=ax, label='QCD+ttbar', color='grey', density=True)
ax.set_xlabel(r'$M_Y$ [GeV]')

ax = axs[4]
bkg.hist(bkg.X.m, x_bins, ax=ax, label='QCD+ttbar', color='grey', density=True)
ax.set_xlabel(r'$M_X$ [GeV]')

fig.savefig(f"{model_savein}/resonance_shapes_bkg.pdf")
# signal = []
# with console.status("[bold][green]Loading signal...") as status:
    # with suppress_stdout():
    # signal = [sixb_from_gnn(out) for out in output]
    # for out in output:
        # try: signal.append(SixB(out))
        # except: print(f"[red]Failed to load: {out}")

    # signal = [sixb_from_gnn(out) for out in get_NMSSM_list()]

sys.exit()

with PdfPages(f"{model_savein}/resonance_shapes_bkg_together.pdf") as pdf:
    # for sig in tqdm(signal):
    for out in tqdm(get_NMSSM_list(run=year)):
        if int(out.split('_')[4].split('-')[1]) > 1200: continue

        try: sig = SixB(out)
        except IndexError: continue

        rprint(f"Processing: mx = {sig.mx}, my = {sig.my}")
        sig.initialize_gen()

        fig, axs = plt.subplots(ncols=5, figsize=(50,10))

        ax = axs[0]
        sig.hist(sig.HX.m, h_bins, ax=ax, label='signal', density=True)
        bkg.hist(bkg.HX.m, h_bins, ax=ax, label='QCD+ttbar', color='grey', density=True)
        ax.set_xlabel(r'$H_X$ Mass [GeV]')
        ax.legend()

        ax = axs[1]
        sig.hist(sig.H1.m, h_bins, ax=ax, label='signal', density=True)
        bkg.hist(bkg.H1.m, h_bins, ax=ax, label='QCD+ttbar', color='grey', density=True)
        ax.set_xlabel(r'$H_1$ Mass [GeV]')
        ax.legend()

        ax = axs[2]
        sig.hist(sig.H2.m, h_bins, ax=ax, label='signal', density=True)
        bkg.hist(bkg.H2.m, h_bins, ax=ax, label='QCD+ttbar', color='grey', density=True)
        ax.set_xlabel(r'$H_2$ Mass [GeV]')
        ax.legend()

        ax = axs[3]
        sig.hist(sig.Y.m, y_bins, ax=ax, label='signal', density=True)
        bkg.hist(bkg.Y.m, y_bins, ax=ax, label='QCD+ttbar', color='grey', density=True)
        ax.set_xlabel(r'$M_Y$ [GeV]')
        ax.legend()

        ax = axs[4]
        sig.hist(sig.X.m, y_bins, ax=ax, label='signal', density=True)
        bkg.hist(bkg.X.m, y_bins, ax=ax, label='QCD+ttbar', color='grey', density=True)
        ax.set_xlabel(r'$M_X$ [GeV]')
        ax.legend()

        pdf.savefig(dpi=300)
        plt.close()
print("DONE")