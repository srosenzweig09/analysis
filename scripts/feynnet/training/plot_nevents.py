import json, re, shutil
import seaborn as sns
from pandas import DataFrame
from utils import *

def human_readable(num):
    num = int(num)
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'k', 'M', 'G', 'T', 'P'][magnitude])

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

# with open("filelists/Summer2018UL/central.txt") as f:
with open("filelists/Summer2018UL/private.txt") as f:
    filelist = f.readlines()

savepath = f"plots/feynnet/"
# tmpdir = f"{savepath}/tmp"
print("Saving to:", savepath)

nevents = {int(mx):{} for mx in MX}
labels = {int(mx):{} for mx in MX}

for file in filelist:
    # for central
    # mxmy = re.search('(MX-.*)_TuneCP5*', file).group(1).replace('-', '_')
    # mx = re.search('MX-(\d+)_', file).group(1)
    # my = re.search('MY-(\d+)_', file).group(1)
    # # for private 
    mxmy = re.search('(MX_.*)/', file).group(1)
    mx = re.search('MX_(\d+)_', file).group(1)
    my = re.search('MY_(\d+)[_/]', file).group(1)

    eff_file = f"tmp/{mxmy}_nevents.json"
    with open(eff_file) as f:
        e = json.load(f)
    
    nevents[int(mx)][int(my)] = e['nevents']
    labels[int(mx)][int(my)] = f"{human_readable(e['nevents'])}"

nevents = get_df(nevents)
labels = get_df(labels)
# nevents.to_csv(f"{savepath}/df.csv")

fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(nevents, cmap='rainbow', ax=ax, annot=labels, fmt='', annot_kws={'fontsize':10})
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel(r"$M_X$ [GeV]")
ax.set_ylabel(r"$M_Y$ [GeV]")
ax.set_title("Number of Training Events")
# turn off minor ticks
# ax.xaxis.set_tick_params(which='minor', bottom=False)
ax.tick_params('both', length=2)
plt.minorticks_off()
fig.savefig(f'{savepath}/training_nevents.pdf', bbox_inches='tight')

# shutil.rmtree(tmpdir)