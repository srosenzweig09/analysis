
print("Loading packages...")
from utils.analysis.gnn import model_path
from utils.analysis import sixb_from_gnn, data_from_gnn, Bkg
from utils.filelists import *
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

asr_mask = False

# read the model path and bdt config file, set up the save directory

model_dir = model_path.split('/')[-2]
savedir = f'plots/feynnet/{model_dir}'
rprint(f"model_path: {model_dir}")
if not os.path.exists(savedir): os.makedirs(savedir)
info_dict['model_path'] = model_dir

cfg = 'config/bdt_params.cfg'
# config = ConfigParser()
# config.optionxform = str
# config.read(cfg)

# if config['spherical']['nregions'] == 'multiple': savedir = f"{savedir}/multiple_bdt_regions"
# elif config['spherical']['nregions'] == 'diagonal': savedir = f"{savedir}/diagonal"
# elif config['spherical']['nregions'] == 'concentric': savedir = f"{savedir}/concentric"

savedir = f"{savedir}/concentric"

if not os.path.exists(savedir): os.makedirs(savedir)
rprint(f"savedir: {savedir}")
console.log(f"[blue]BDT region:[\blue] {savedir.split('/')[-1]}")
info_dict['bdt_region'] = savedir.split('/')[-1]

print()
console.log(info_dict)
print()

# if not os.path.exists(savedir + '/root_files'): os.makedirs(savedir + '/root_files')

# load the data and train the background model

# data = data_from_gnn('/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/JetHT_Data_UL/ntuple.root', config=cfg)

# print("Data loaded.")
# print("Training data...")

# data.spherical_region()
# data.train()

# fig, ax, n_4b_model = data.sr_hist(savein=f"{savedir}/data_asr_model.root")
# print(n_4b_model)
# ax[0].set_title('GNN, w/ MX reweighting')
# fig.savefig(f'{savedir}/sr_model.pdf')

# data.pull_plots(savein=savedir, filename='gnn_pull')

# load the mc bkg and train the background model
print("Loading mc qcd + ttbar...")
bkg = Bkg(get_qcd_ttbar('maxbtag_4b'), gnn=True)
bkg.spherical_region(cfg=cfg, nregions='concentric')

# load the signal
base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM'
cmd = f"ls {base}"
output = subprocess.check_output(shlex.split(cmd))
output = output.decode('UTF-8')
output = output.split('\n')
output = [f"{base}/{out}/ntuple.root" for out in output if out.startswith('NMSSM')]

with console.status("[bold][green]Loading signal...") as status:
    # with suppress_stdout():
    signal = [sixb_from_gnn(out) for out in output]
    [sig.spherical_region(nregions='concentric') for sig in signal]

def get_figure(sig):
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(40,8))
    mx, my = sig.mx, sig.my

    mask = sig.resolved_mask
    if asr_mask: mask = mask & sig.asr_mask

    n = sig.hist(sig.HX.m[mask], bins=np.linspace(0,400,31), ax=axs[0], density=True, label=f"[{mx},{my}]")
    n = bkg.hist(bkg.HX.m, bins=np.linspace(0,400,31), mask=bkg.asr_mask, plot_mode='separate', ax=axs[0], label='qcd+ttbar', density=True)
    axs[0].set_xlabel(r'$m_{H_X}$ [GeV]')

    n = sig.hist(sig.H1.m[mask], bins=np.linspace(0,400,31), ax=axs[1], density=True, label=f"[{mx},{my}]")
    n = bkg.hist(bkg.H1.m, bins=np.linspace(0,400,31), mask=bkg.asr_mask, plot_mode='separate', ax=axs[1], label='qcd+ttbar', density=True)
    axs[1].set_xlabel(r'$m_{H_1}$ [GeV]')

    n = sig.hist(sig.H2.m[mask], bins=np.linspace(0,400,31), ax=axs[2], density=True, label=f"[{mx},{my}]")
    n = bkg.hist(bkg.H2.m, bins=np.linspace(0,400,31), mask=bkg.asr_mask, plot_mode='separate', ax=axs[2], label='qcd+ttbar', density=True)
    axs[2].set_xlabel(r'$m_{H_2}$ [GeV]')

    n = sig.hist(sig.Y.m[mask], bins=np.linspace(200,1100,31), ax=axs[3], density=True, label=f"[{mx},{my}]")
    n = bkg.hist(bkg.Y.m, bins=np.linspace(200,1100,31), mask=bkg.asr_mask, plot_mode='separate', ax=axs[3], label='qcd+ttbar', density=True)
    axs[3].set_xlabel(r'$m_{Y}$ [GeV]')
    
    n = sig.hist(sig.X.m[mask], bins=np.linspace(375,2000,31), ax=axs[4], density=True, label=f"[{mx},{my}]")
    n = bkg.hist(bkg.X.m, bins=np.linspace(375,2000,31), mask=bkg.asr_mask, plot_mode='separate', ax=axs[4], label='qcd+ttbar', density=True)
    axs[4].set_xlabel(r'$m_{Y}$ [GeV]')

    for ax in axs.flatten():
        ax.legend()

    return fig, axs

filename = f"{savedir}/mh_resolved"
if asr_mask: filename = f"{filename}_asr"
print("..generating resolved mh, my plots")
with PdfPages(f"{filename}.pdf") as pdf:
    for sig in tqdm(signal):
        rprint(f"Processing: mx = {sig.mx}, my = {sig.my}")
        fig, axs = get_figure(sig)
        pdf.savefig(dpi=300)
        plt.close()

rprint(f".. saved in {filename}.pdf")
print("DONE")