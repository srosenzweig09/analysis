from matplotlib.backends.backend_pdf import PdfPages
from utils.analysis import sixb_from_gnn, Bkg
import matplotlib.pyplot as plt
from rich import print as rprint
from rich.console import Console
console = Console()
from tqdm import tqdm
from utils.plotter import Hist2d
import numpy as np
from utils.filelists import *
from matplotlib.patches import Circle
import os

from utils.analysis.gnn import model_path
model_dir = model_path.split('/')[-2]
model_savein = f'plots/feynnet/{model_dir}'
model_dir
if not os.path.exists(model_savein): os.makedirs(model_savein)

info_dict = {}

nbins = 31
h_bins = np.linspace(0,300,nbins)
y_bins = np.linspace(0,2000,nbins)
x_bins = np.linspace(0,2200,nbins)

ar_color = 'black'
vr_color = 'white'
cr_color = 'violet'

bkg = Bkg(get_qcd_ttbar(selection='maxbtag_4b'))

fig, axs = plt.subplots(ncols=2, figsize=(20,8))

ax = axs[0]
ax.set_title('QCD+ttbar')
n, ex, ey, im = Hist2d(bkg.HX.m, bkg.H1.m, bins=np.linspace(0,350,100), ax=ax)
ax.set_xlabel(r'$H_X$ Mass [GeV]')
ax.set_ylabel(r'$H_1$ Mass [GeV]')
fig.colorbar(im, ax=ax)

arcircle1 = Circle((125,125), radius=25, color=ar_color, fill=False, lw=2)
vrcircle1 = Circle((125,125), radius=50, color=vr_color, fill=False, lw=2)
crcircle1 = Circle((125,125), radius=75, color=cr_color, fill=False, lw=2)

ax.add_artist(arcircle1)
ax.add_artist(vrcircle1)
ax.add_artist(crcircle1)

ax = axs[1]
n, ex, ey, im = Hist2d(bkg.H1.m, bkg.H2.m, bins=np.linspace(0,350,100), ax=ax)
ax.set_xlabel(r'$H_1$ Mass [GeV]')
ax.set_ylabel(r'$H_2$ Mass [GeV]')
fig.colorbar(im, ax=ax)

arcircle2 = Circle((125,125), radius=25, color=ar_color, fill=False, lw=2)
vrcircle2 = Circle((125,125), radius=50, color=vr_color, fill=False, lw=2)
crcircle2 = Circle((125,125), radius=75, color=cr_color, fill=False, lw=2)

ax.add_artist(arcircle2)
ax.add_artist(vrcircle2)
ax.add_artist(crcircle2)



# signal = []
# with console.status("[bold][green]Loading signal...") as status:
    # with suppress_stdout():
    # signal = [sixb_from_gnn(out) for out in output]
    # for out in output:
        # try: signal.append(SixB(out))
        # except: print(f"[red]Failed to load: {out}")

    # signal = [sixb_from_gnn(out) for out in get_NMSSM_list()]


with PdfPages(f"{model_savein}/higgs_mass_2d.pdf") as pdf:
    # for sig in tqdm(signal):
    for out in tqdm(get_NMSSM_list()):
        sig = sixb_from_gnn(out)
        rprint(f"Processing: mx = {sig.mx}, my = {sig.my}")
        sig.initialize_gen()

        fig, axs = plt.subplots(ncols=2, figsize=(20,8))

        ax = axs[0]
        ax.set_title(f"NMSSM Signal: $m_X$ = {sig.mx}, $m_Y$ = {sig.my}")
        n, ex, ey, im = Hist2d(sig.HX.m, sig.H1.m, bins=np.linspace(0,350,100), ax=ax)
        ax.set_xlabel(r'$H_X$ Mass [GeV]')
        ax.set_ylabel(r'$H_1$ Mass [GeV]')
        fig.colorbar(im, ax=ax)

        arcircle1 = Circle((125,125), radius=25, color=ar_color, fill=False, lw=2)
        vrcircle1 = Circle((125,125), radius=50, color=vr_color, fill=False, lw=2)
        crcircle1 = Circle((125,125), radius=75, color=cr_color, fill=False, lw=2)

        ax.add_artist(arcircle1)
        ax.add_artist(vrcircle1)
        ax.add_artist(crcircle1)

        ax = axs[1]
        n, ex, ey, im = Hist2d(sig.H1.m, sig.H2.m, bins=np.linspace(0,350,100), ax=ax)
        ax.set_xlabel(r'$H_1$ Mass [GeV]')
        ax.set_ylabel(r'$H_2$ Mass [GeV]')
        fig.colorbar(im, ax=ax)

        arcircle2 = Circle((125,125), radius=25, color=ar_color, fill=False, lw=2)
        vrcircle2 = Circle((125,125), radius=50, color=vr_color, fill=False, lw=2)
        crcircle2 = Circle((125,125), radius=75, color=cr_color, fill=False, lw=2)

        ax.add_artist(arcircle2)
        ax.add_artist(vrcircle2)
        ax.add_artist(crcircle2)

        pdf.savefig(dpi=300)
        plt.close()
print("DONE")