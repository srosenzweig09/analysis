# sbatch scripts/feynnet/efficiencies/run

import json, re, shutil
import seaborn as sns
from pandas import DataFrame
from utils import *
from utils.analysis.feyn import Model
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", default="config/feynnet.cfg")
args = parser.parse_args()

rand_num = re.search("feynnet_(.*).cfg", args.config).group(1)

model_version, model_name, model_path = Model.read_feynnet_cfg(args.config)

def get_mx_my(mass):
    mx = int(mass.split('/')[-2].split('_')[2].split('-')[1])
    my = int(mass.split('/')[-2].split('_')[3].split('-')[1])
    return mx, my

def get_df(var):
    df = DataFrame.from_dict(var)
    df = df.reindex(index=df.index[::-1])
    return df

mx_my_masses = [get_mx_my(mass) for mass in get_NMSSM_list()]
mx_my_masses = [[mx,my] for mx,my in mx_my_masses if mx < 1300]
MX = np.unique(np.array(mx_my_masses)[:,0])

with open("filelists/Summer2018UL/central.txt") as f:
# with open("filelists/Summer2018UL/private.txt") as f:
    filelist = f.readlines()

savepath = f"plots/feynnet/{model_name}"
if not os.path.exists(savepath): os.makedirs(savepath)
# tmpdir = f"{savepath}/tmp"
print("Saving to:", savepath)

eff = {int(mx):{} for mx in MX}
res = {int(mx):{} for mx in MX}
sr = {int(mx):{} for mx in MX}
reco = {int(mx):{} for mx in MX}

for file in filelist:
    # for central
    mxmy = re.search('(MX-.*)_TuneCP5*', file).group(1).replace('-', '_')
    mx = re.search('MX-(\d+)_', file).group(1)
    my = re.search('MY-(\d+)_', file).group(1)
    # # for private 
    # mxmy = re.search('(MX_.*)/', file).group(1)
    # mx = re.search('MX_(\d+)_', file).group(1)
    # my = re.search('MY_(\d+)_', file).group(1)

    eff_file = f"tmp/{rand_num}_{mxmy}_efficiency.json"
    with open(eff_file) as f:
        e = json.load(f)
    
    eff[int(mx)][int(my)] = e['feynnet_efficiency']
    res[int(mx)][int(my)] = e['resolved_efficiency']
    sr[int(mx)][int(my)] = e['sr_efficiency']
    reco[int(mx)][int(my)] = e['n_possible_higgs']

eff = get_df(eff)
eff.to_csv(f"{savepath}/df.csv")

res = get_df(res)
sr = get_df(sr)
reco = get_df(reco)

fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(eff, cmap='rainbow', ax=ax, vmin=0.0, vmax=1.05)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel(r"$M_X$ [GeV]")
ax.set_ylabel(r"$M_Y$ [GeV]")
ax.set_title("FeynNet Reconstruction Efficiency")
fig.savefig(f'{savepath}/feynnet_efficiency.pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(res, cmap='rainbow', ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel(r"$M_X$ [GeV]")
ax.set_ylabel(r"$M_Y$ [GeV]")
ax.set_title("Resolved Rate")
fig.savefig(f'{savepath}/resolved_rate.pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(sr, cmap='rainbow', ax=ax, vmin=0.0, vmax=0.2)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel(r"$M_X$ [GeV]")
ax.set_ylabel(r"$M_Y$ [GeV]")
ax.set_title("Signal Region Efficiency")
fig.savefig(f'{savepath}/sr_efficiency.pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(reco, cmap='rainbow', ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel(r"$M_X$ [GeV]")
ax.set_ylabel(r"$M_Y$ [GeV]")
ax.set_title("Average Number of Gen-Matched Higgs Pairs")
fig.savefig(f'{savepath}/h_possible.pdf', bbox_inches='tight')

# shutil.rmtree(tmpdir)