# sh scripts/parallel/runPythonWithFilelist.sh scripts/background_estimation/sr_cr_optimization.py filelists/central.txt
# python scripts/background_estimation/sr_cr_optimization.py /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/NMSSM_XToYHTo6B_MX-1000_MY-250_TuneCP5_13TeV-madgraph-pythia8/ntuple.root

"""
This script will generate plots of the reconstructed Higgs masses, as well as the average b tag score of signal samples.
"""

from utils import *

parser = ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

def raiseError(fname):
    print(f"{Fore.RED}[FAILED]{Style.RESET_ALL} Cannot open file:\n{fname}")
    raise

fname = args.filename
savedir = 'plots/background_estimation'

try: tree = SixB(fname)
except: raiseError(fname)
bins = np.linspace(50,250,41)

# Individual reco Higgs masses #####################################################
fig, axs = plt.subplots(ncols=3, figsize=(30,8))

n = Hist(tree.HX.m, bins=bins, label='HX', ax=axs[0], weights=tree.nomWeight)
n = Hist(tree.H1.m, bins=bins, label='H1', ax=axs[1], weights=tree.nomWeight)
n = Hist(tree.H2.m, bins=bins, label='H2', ax=axs[2], weights=tree.nomWeight)

for i,ax in enumerate(axs):
    ax.set_title(tree.sample)
    ax.set_xlabel(f'Reco Higgs {i+1} Mass [GeV]')
    ax.set_ylabel('Events')

fig.savefig(f'{savedir}/{tree.mxmy}_recoHiggsMass.pdf', bbox_inches='tight')

# 2D reco Higgs masses #############################################################
fig, axs = plt.subplots(ncols=2, figsize=(20,10))

n, im1 = Hist2d(tree.HX.m, tree.H1.m, bins=bins, ax=axs[0], weights=tree.nomWeight)
axs[0].set_xlabel(f'H1 Reco Mass [GeV]')
axs[0].set_ylabel(f'H2 Reco Mass [GeV]')
n, im2 = Hist2d(tree.H1.m, tree.H2.m, bins=bins, ax=axs[1], weights=tree.nomWeight)
axs[1].set_xlabel(f'H2 Reco Mass [GeV]')
axs[1].set_ylabel(f'H3 Reco Mass [GeV]')

fig.colorbar(im1, ax=axs[0])
fig.colorbar(im2, ax=axs[1])
for i,ax in enumerate(axs):
    ax.set_title(tree.sample)

fig.savefig(f'{savedir}/{tree.mxmy}_recoHiggsMass2D.pdf', bbox_inches='tight')

# Average b tag score ###############################################################
fig, ax = plt.subplots(figsize=(8,8))

n = Hist(tree.btag_avg, bins=np.linspace(0,1,41), ax=ax, weights=tree.nomWeight, label='All Events')
n = Hist(tree.btag_avg[tree.ar_mask], bins=np.linspace(0,1,41), ax=ax, weights=tree.nomWeight[tree.ar_mask], label='Analysis Region')
ax.set_title(tree.sample)
ax.set_xlabel('Average b tag score')
ax.set_ylabel('Events')

fig.savefig(f'{savedir}/{tree.mxmy}_btag_avg.pdf', bbox_inches='tight')