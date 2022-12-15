"""
The goal for this script is to compare various event selections by building a model of the background and comparing the limits for each method.

* It needs to 
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# from configparser import ConfigParser
from utils.analysis.signal import SixB, Data
from utils.bkg_model import Model
from utils.plotter import Hist

import matplotlib.pyplot as plt
import subprocess, shlex
import sys

xsec_factor = 300
# xsec_factor = 300 / (0.7 * 0.58**3)

file_location = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/btag_pt'
cfg_file = 'config/bdt_params.cfg'

# config = ConfigParser()
# config.optionxform = str
# config.read(cfg_file)

data_fname = f"{file_location}/JetHT_Data_UL/JetHT_Run2018_full/ntuple.root"
data = Data(data_fname, cfg_file)

data.train()
# _ = data.v_cr_hist()
# _ = data.v_sr_hist()
fig, ax, n_data = data.sr_hist()
plt.close(fig)

limit_list = ['mx my -2sigma -1sigma mean +1sigma +2sigma']

print(".. calculating limits")
cmd = f'ls {file_location}/NMSSM'
output = subprocess.check_output(shlex.split(cmd))
signal_dirs = output.decode('UTF-8').split('\n')
for i,signal_dir in enumerate(signal_dirs):
    if 'analysis_tar' in signal_dir: continue
    if len(signal_dir) == 0: continue
    # if i > 2: continue
    print(".. processing",signal_dir)
    signal_tree = SixB(f"{file_location}/NMSSM/{signal_dir}/ntuple.root", cfg_file)
    cutflow = signal_tree.cutflow_norm

    fig, ax, n_sig = signal_tree.sr_hist()
    Hist(data.x_mBins, bins=data.mBins, weights=n_data, ax=ax, label='Expected Background', zorder=1)
    fig.savefig(f'limits/btag_pt/signal/{signal_tree.mxmy}.pdf')
    plt.close(fig)

    model = Model(n_sig, n_data, sumw2=data.sumw2)
    limits = model.upperlimit()
    limits = limits[1]
    limits = [str(round(limit.to_py().tolist()*xsec_factor, 1)) for limit in limits]

    limits = [str(signal_tree.mx), str(signal_tree.my)] + limits# + ['\n']
    limit_list.append(' '.join(limits))
    # sys.exit()


limit_list[-1] = limit_list[-1] + '\n'
print(limit_list)
print('\n'.join(limit_list))

print(".. writing limits to txt file")
with open("limits/btag_pt_limits.txt", "w") as f:
    f.write('\n'.join(limit_list))



