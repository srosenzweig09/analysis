from utils import *
import utils.analysis.signal as uas
import matplotlib.pyplot as plt
import os

mpoints = [
    [450,300],
    [700,500],
    [1000,300],
    [1000,600],
    [1200,300],
    [1200,1000]
]

density = False

# bkg = Bkg(get_qcd_ttbar('maxbtag_4b'))
bkg = uas.Bkg()
# bkg.spherical_region()
signals = [SixB(get_NMSSM(*m)) for m in mpoints]

savepath = signals[0].model.savepath
if not os.path.exists(savepath): os.makedirs(savepath)
print(savepath)

fig, axs = plt.subplots(ncols=3, nrows=len(signals), figsize=(30,8*len(signals)))

qcd_color = 'wheat'
ttbar_color = 'lightblue'
bins = np.linspace(0,300,41)
xlabels = [f'Higgs {i+1} Mass [GeV]' for i in range(3)]

for i,signal in enumerate(signals):
    ax = axs[i,0]
    n = Hist(bkg.qcd.H1_m, bins=bins, ax=ax, weights=bkg.qcd.nomWeight, label='qcd', density=density, histtype='stepfilled', color=qcd_color)
    n = Hist(bkg.ttbar.H1.m, bins=bins, ax=ax, weights=bkg.ttbar.nomWeight, bottom=n, label='ttbar', density=density, histtype='stepfilled', color=ttbar_color)
    n = Hist(signal.H1.m, bins=bins, ax=ax, weights=signal.nomWeight, density=density, bottom=n, label=signal.sample)
    
    ax = axs[i,1]
    n = Hist(bkg.qcd.H2_m, bins=bins, ax=ax, weights=bkg.qcd.nomWeight, density=density, label='qcd', histtype='stepfilled', color=qcd_color)
    n = Hist(bkg.ttbar.H2.m, bins=bins, ax=ax, weights=bkg.ttbar.nomWeight, density=density, bottom=n, label='ttbar', histtype='stepfilled', color=ttbar_color)
    n = Hist(signal.H2.m, bins=bins, ax=ax, weights=signal.nomWeight, density=density, bottom=n, label=signal.sample)
    
    ax = axs[i,2]
    n = Hist(bkg.qcd.H3_m, bins=bins, ax=ax, weights=bkg.qcd.nomWeight, density=density, label='qcd', histtype='stepfilled', color=qcd_color)
    n = Hist(bkg.ttbar.H3.m, bins=bins, ax=ax, weights=bkg.ttbar.nomWeight, density=density, bottom=n, label='ttbar', histtype='stepfilled', color=ttbar_color)
    n = Hist(signal.H3.m, bins=bins, ax=ax, weights=signal.nomWeight, density=density, bottom=n, label=signal.sample)

    for j,ax in enumerate(axs[i]):
        ax.set_title(signal.sample)
        ax.set_xlabel(xlabels[j])
        ax.set_ylabel('AU')
        ax.legend()

fig.suptitle(bkg.ttbar.model_name, y=1.0)
plt.tight_layout()
fig.savefig(f'{savepath}/reco_higgs_masses.pdf', bbox_inches='tight')


fig, axs = plt.subplots(nrows=len(signals)+2, figsize=(10,8*(len(signals)+2)))

bins = np.linspace(0,300,41)
for i,tree in enumerate([bkg.qcd, bkg.ttbar] + signals):

    try: hx,h1,h2 = tree.H1.m, tree.H2.m, tree.H3.m
    except: hx,h1,h2 = tree.H1_m, tree.H2_m, tree.H3_m
    if not isinstance(hx, np.ndarray): hx = hx.to_numpy()
    if not isinstance(h1, np.ndarray): h1 = h1.to_numpy()
    if not isinstance(h2, np.ndarray): h2 = h2.to_numpy()
    weight = tree.nomWeight.to_numpy()

    ax = axs[i]

    n, im = Hist2d(hx, h1, bins=bins, ax=ax, weights=weight)

    ax.set_title(tree.sample)
    fig.colorbar(im, ax=ax)

fig.suptitle(bkg.ttbar.model_name, y=1.0)
plt.tight_layout()
fig.savefig(f'{savepath}/reco_higgs_masses_2d.pdf', bbox_inches='tight')


fig, axs = plt.subplots(nrows=len(signals), figsize=(8,6*len(signals)))

bins = np.linspace(0,1,61)
for i,signal in enumerate(signals):

    ax = axs[i]
    n = Hist(bkg.qcd.btag_avg[bkg.qcd.asr_mask], bins=bins, ax=ax, density=density, label='qcd', histtype='stepfilled', color=qcd_color)

    n = Hist(bkg.ttbar.btag_avg[bkg.ttbar.asr_mask], bins=bins, ax=ax, density=density, bottom=n, label='ttbar', histtype='stepfilled', color=ttbar_color)

    n = Hist(signal.btag_avg[signal.asr_mask], bins=bins, ax=ax, density=density, bottom=n, label=signal.sample)
    
    ax.set_title(signal.sample)
    ax.set_xlabel(xlabels[j])
    ax.set_ylabel('AU')
    ax.legend(loc=2)

fig.suptitle(bkg.ttbar.model_name, y=1.0)
plt.tight_layout()
fig.savefig(f'{savepath}/avg_btag.pdf', bbox_inches='tight')