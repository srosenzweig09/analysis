# python scripts/weights_and_scalefactors/check_weights/plot_weights.py

import json, re, shutil
from utils import *
from utils.analysis.feyn import Model
from argparse import ArgumentParser
from matplotlib.backends.backend_pdf import PdfPages

parser = ArgumentParser()
parser.add_argument("--config", default="config/feynnet.cfg")
args = parser.parse_args()

model_version, model_name, model_path = Model.read_feynnet_cfg(args.config)

def get_mx_my(mass):
    mx = int(mass.split('/')[-2].split('_')[2].split('-')[1])
    my = int(mass.split('/')[-2].split('_')[3].split('-')[1])
    return mx, my

mx_my_masses = [get_mx_my(mass) for mass in get_NMSSM_list()]
mx_my_masses = [[mx,my] for mx,my in mx_my_masses if mx < 1300]
MX = np.unique(np.array(mx_my_masses)[:,0])

with open("filelists/Summer2018UL/central.txt") as f:
    filelist = f.readlines()
filelist = natural_sort(filelist)


savepath = f"plots"
tmpdir = f"{savepath}/tmp"

with PdfPages(f'{savepath}/weights.pdf') as pdf:
    for file in filelist:
        mxmy = re.search('(MX-.*)_TuneCP5*', file).group(1).replace('-', '_')
        mx = re.search('MX-(\d+)_', file).group(1)
        my = re.search('MY-(\d+)_', file).group(1)

        weights = np.load(f"{tmpdir}/{mxmy}_weights.npz")

        X_m = weights['X_m']
        genWeight = weights['genWeight']
        w_pu = weights['w_pu']
        w_pu_up = weights['w_pu_up']
        w_pu_down = weights['w_pu_down']
        w_puid = weights['w_puid']
        w_puid_up = weights['w_puid_up']
        w_puid_down = weights['w_puid_down']
        w_trigger = weights['w_trigger']
        w_trigger_up = weights['w_trigger_up']
        w_trigger_down = weights['w_trigger_down']

        fig, axs = plt.subplots(ncols=3, figsize=(30,8))

        n = Hist(X_m, weights=genWeight*w_pu, bins=np.linspace(375,1500,41), ax=axs[0], label='PU')
        n = Hist(X_m, weights=genWeight*w_pu_up, bins=np.linspace(375,1500,41), ax=axs[0], label='PU up')
        n = Hist(X_m, weights=genWeight*w_pu_down, bins=np.linspace(375,1500,41), ax=axs[0], label='PU down')

        n = Hist(X_m, weights=genWeight*w_puid, bins=np.linspace(375,1500,41), ax=axs[1], label='PUID')
        n = Hist(X_m, weights=genWeight*w_puid_up, bins=np.linspace(375,1500,41), ax=axs[1], label='PUID up')
        n = Hist(X_m, weights=genWeight*w_puid_down, bins=np.linspace(375,1500,41), ax=axs[1], label='PUID down')

        n = Hist(X_m, weights=genWeight*w_trigger, bins=np.linspace(375,1500,41), ax=axs[2], label='Trigger')
        n = Hist(X_m, weights=genWeight*w_trigger_up, bins=np.linspace(375,1500,41), ax=axs[2], label='Trigger up')
        n = Hist(X_m, weights=genWeight*w_trigger_down, bins=np.linspace(375,1500,41), ax=axs[2], label='Trigger down')

        for ax in axs:
            ax.set_title(mxmy)
            ax.set_xlabel(r'Reco $M_X$ [GeV]')
            ax.set_ylabel('Events')
            ax.legend()

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

shutil.rmtree(tmpdir)