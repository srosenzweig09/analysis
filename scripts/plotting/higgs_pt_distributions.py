import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import subprocess
import sys

from utils.analysis.signal import SixB
from utils.useCMSstyle import *
plt.style.use(CMS)
from utils.plotter import Hist

base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/btag_pt/NMSSM'
cmd = f'ls {base}'

output = subprocess.check_output(cmd.split(' '))
output = output.decode('UTF-8').split('\n')
output.remove('analysis_tar')
output.remove('')

columns = ['MX', 'MY', 'None', 'Trigger', 'Preselections', 'Selections']
efficiencies = []

pt_bins = np.linspace(0, 700, 60)
fig, ax = plt.subplots()

for i,subdir in enumerate(output):
    print(subdir)

    subdir = f'{base}/{subdir}'
    
    tree = SixB(f'{subdir}/ntuple.root')
    # print(tree.cutflow_norm, tree.cutflow_scaled)
    mx, my = tree.mx, tree.my
    
    efficiencies.append([mx, my] + tree.cutflow_norm.tolist())

    ax.clear()
    Hist(tree.HX_pt, bins=pt_bins, weights=tree.scale, ax=ax, label=r'$H_X$ Candidate')
    Hist(tree.H1_pt, bins=pt_bins, weights=tree.scale, ax=ax, label=r'$H_1$ Candidate')
    Hist(tree.H2_pt, bins=pt_bins, weights=tree.scale, ax=ax, label=r'$H_2$ Candidate')

    ax.set_title(tree.sample)
    ax.set_xlabel(r'$p_T$ [GeV]')
    ax.set_ylabel('AU')

    fig.savefig(f'plots/topology/{tree.mxmy}')

    # plt.show()
    # sys.exit()

df = DataFrame(efficiencies, columns=columns)

df = df.astype({key:int for key in columns[:2]})
df = df.set_index(columns[:2])
df = df.sort_index()

df.to_csv('plots/topology/efficiencies.csv')