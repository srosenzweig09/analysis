from rich.console import Console
console = Console()

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from utils.analysis import sixb_from_gnn
from utils.plotter import r_cmap
import subprocess, shlex

from utils.analysis.gnn import model_path
model_dir = model_path.split('/')[-2]
model_savein = f'plots/feynnet/{model_dir}'
model_dir, model_savein
console.log(f"[blue]model_path:[/blue] {model_dir}")

console.log('Reading config...')
base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM'
cmd = f"ls {base}"
output = subprocess.check_output(shlex.split(cmd))
output = output.decode('UTF-8')
output = output.split('\n')
signal_list = [f"{base}/{out}/ntuple.root" for out in output if out.startswith('NMSSM')]


with console.status("[bold][green]Loading signal...") as status:
    gnn_signal = [sixb_from_gnn(out) for out in signal_list if int(out.split('-')[1].split('_')[0]) % 100 == 0]
console.log('Signal loaded!')

title = '26 training masses, 100 epochs'

console.log('Obtaining efficiencies...')

X, Y, Z = [], [], []

for signal in gnn_signal:
    # sig_eff[signal.mx][signal.my] = ak.sum(signal.n_H_correct[signal.resolved_mask]==3)/ak.sum(signal.resolved_mask)
    X.append(signal.mx)
    Y.append(signal.my)
    Z.append(ak.sum(signal.n_H_correct[signal.resolved_mask]==3)/ak.sum(signal.resolved_mask))

X = np.asarray(X)
Y = np.asarray(Y)
Z = np.asarray(Z)

console.log('Interpolating efficiencies...')

from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator

interp_kind = dict(
        linear=LinearNDInterpolator,
        clough=CloughTocher2DInterpolator,
    ).get('linear', LinearNDInterpolator)

# f = interp_kind(np.array([X,Y]).T, Z)
f = LinearNDInterpolator(np.array([X,Y]).T, Z)
nx = np.linspace(np.min(X), np.max(X), 100)
ny = np.linspace(np.min(Y), np.max(Y), 100)
nx, ny = np.meshgrid(nx, ny)
nz = f(nx, ny)

MX = [450,550,600,700,800,800,900,900,1000,1000,1100,1100,1100,1000,1000,1100,1200,1200,900,900,800,800,700,600,500,400]
MY = [300,300,350,400,350,500,300,500,400 ,600 ,300 ,500 ,700 ,250 ,800 ,250 ,250 ,1000,250,700,250,600,250,250,250,250]

console.log('Plotting efficiencies...')

fig, ax = plt.subplots(figsize=(10,8))

im = ax.pcolor(nx, ny, nz, cmap=r_cmap, vmin=0.0, vmax=1.0)
fig.colorbar(im, ax=ax, label='Efficiency')

ax.scatter(MX, MY, color='yellow', zorder=2)
ax.scatter(X,Y,color='k')
# ax.scatter([450,550,700,900,1000],[300,300,400,300,600], color='yellow', zorder=2)

ax.set_xlim(350,1250)
ax.set_ylim(225,1050)

ax.set_xlabel(r'$M_X$ [GeV]')
ax.set_ylabel(r'$M_Y$ [GeV]')

ax.set_title(title)

fig.savefig(f'{model_savein}/reco_eff.pdf')

console.log(f'Efficiencies saved in {model_savein}/reco_eff.pdf')