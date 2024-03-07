# scripts/feynnet/ranking/plot_ranks.py

import json, re, shutil, os
import seaborn as sns
from pandas import DataFrame
from utils import *
from utils.analysis.feyn import Model
from argparse import ArgumentParser
import glob
from matplotlib.backends.backend_pdf import PdfPages
from utils.plotter import get_cmap_kwargs

parser = ArgumentParser()
parser.add_argument("--config", default="config/feynnet.cfg")
args = parser.parse_args()

rand_num = re.search("feynnet_(.*).cfg", args.cfg).group(1)

model_version, model_name, model_path = Model.read_feynnet_cfg(args.config)

def get_mx_my(mass):
    mx = int(mass.split('/')[-2].split('_')[2].split('-')[1])
    my = int(mass.split('/')[-2].split('_')[3].split('-')[1])
    return mx, my

def get_df(var):
    df = DataFrame.from_dict(var)
    df = df.reindex(index=df.index[::-1])
    return df

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

mx_my_masses = [get_mx_my(mass) for mass in get_NMSSM_list()]
mx_my_masses = [[mx,my] for mx,my in mx_my_masses if mx < 1300]
MX = np.unique(np.array(mx_my_masses)[:,0])

savepath = f"plots/feynnet/{model_name}"
# tmpdir = f"{savepath}/tmp"
files = glob.glob(f"tmp/{rand_num}*ranking.npy")
sorted_files = natural_sort(files)

ranks = {}
maxrank = {int(mx):{} for mx in MX}
frac = {int(mx):{} for mx in MX}
avg = {int(mx):{} for mx in MX}

with PdfPages(f'{savepath}/feynnet_ranks.pdf') as pdf:
    for file in sorted_files:
        mx = int(re.search('MX_(\d+)_', file).group(1))
        my = int(re.search('MY_(\d+)_', file).group(1))

        tmp_ranks = np.load(file)
        unique, counts = np.unique(tmp_ranks, return_counts=True)
        maxrank[mx][my] = unique[np.argmax(counts)]
        frac[mx][my] = np.max(counts)/np.sum(counts)
        
        avg[mx][my] = np.average(tmp_ranks)

        fig, ax = plt.subplots(figsize=(15,8))
        mxmy = re.search('(MX_.*_MY_.*)_', file).group(1)
        n = Hist(tmp_ranks, bins=np.arange(18), ax=ax, label=mxmy, density=True)
        ax.legend()
        ax.set_xlabel('Correct FeynNet Rank')
        ax.set_ylabel('Fraction of Events')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

df = get_df(maxrank)
frac = get_df(frac)
avg = get_df(avg)
cmap_norm = get_cmap_kwargs(np.arange(1,16), 'rainbow')

fig, ax = plt.subplots(figsize=(10,8))
hm = sns.heatmap(avg, ax=ax, vmin=1, vmax=10, annot=True, annot_kws={'fontsize':10}, **cmap_norm)
ax.set_xlabel('gen MX')
ax.set_ylabel('gen MY')
ax.set_title('Average Rank of Correct FeynNet Combination')
fig.savefig(f'{savepath}/feynnet_ranks_heatmap.pdf', bbox_inches='tight')


fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df, ax=ax, vmin=1, vmax=16, annot=frac, annot_kws={'fontsize' : 10}, **cmap_norm)
ax.set_xlabel('gen MX')
ax.set_ylabel('gen MY')

ax.set_title('Most Frequent Rank of Correct FeynNet Combination')
fig.savefig(f'{savepath}/correct_ranks_heatmap.pdf', bbox_inches='tight')

# shutil.rmtree(tmpdir)
# os.remove(f"tmp/{rand_num}*ranking.npy")