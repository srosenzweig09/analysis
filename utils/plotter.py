"""
I would like my plots to be consistent so I've saved what I can in the rcparams file but there are some things I cannot change. This script will help me keep those changes consistent.
"""

from . import *

# from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import numpy as np

file_location = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_2_18/src/sixb/plots/'

easy_bins = {
    'pt' : np.linspace(0, 300, 100),
    'eta' : np.linspace(-3, 3, 100)
}
easy_labels = {
    'pt' : r'jet $p_T$ [GeV]'
}

legend_loc = {
    'best' : 0,
    'upper right' : 1,
    'upper left' : 2,
    'lower left' : 3,
    'lower right' : 4,
    'right' : 5,
    'center left' : 6,
    'center right' : 7,
    'lower center' : 8,
    'upper center' : 9,
    'center' : 10 }



def change_cmap_bkg_to_white(colormap, n=256):
    """The lowest value of colormaps is not often white by default, which can help idenfity empty bins.
    This function will make the lowest value (typically zero) white."""
    
    tmp_colors = cm.get_cmap(colormap, n)
    newcolors = tmp_colors(np.linspace(0, 1, n))
    white = np.array([1, 1, 1, 1])    # White background (Red, Green, Blue, Alpha).
    newcolors[0, :] = white    # Only change bins with 0 entries.
    newcmp = colors.ListedColormap(newcolors)
    
    return newcmp

class Plotter:
    def __init__():
        pass

def hist(x, bins=100, label=None, weights=None, color=None, density=False, stacked=False, histtype='step', alpha=1.0, fig=None, ax=None, xlim=None, ylim=None, xlabel=None, ylabel=None, savefig=None, pdf=False, title=None, useMathText=False):
    # try: bins = np.linspace(x.min(), x.max(), bins)
    # except: pass
    if ax is None: fig, ax = plt.subplots()
    if not (weights is None): x = (bins[1:] + bins[:-1])/2
    # try: bins = np.linspace(x.min(), x.max(), bins)
    # except: pass
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)
    N = len(x)
    if weights is not None: N = sum(weights)
    textstr = f'Entries = {N}'
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax.text(1.0, 1.0, textstr, transform=ax.transAxes, fontsize=9,
        va='bottom', bbox=props, ha='right')
    ax.hist(x, bins=bins, histtype=histtype, align='mid', label=label, weights=weights,  color=color, density=density, stacked=stacked, alpha=alpha)
    suffix = ''
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if pdf: suffix = '.pdf'
    if label is not None: ax.legend()
    if savefig is not None: fig.savefig(file_location + savefig + suffix)
    plt.tight_layout()

def hist2d(ax, x, y, xbins=100, ybins=100, norm=clrs.LogNorm()):
    cmap = change_cmap_bkg_to_white('rainbow')
    return ax.hist2d(x, y, bins=(xbins, ybins), norm=clrs.LogNorm(), cmap=cmap)

def norm_hist(arr, bins=100):
    n, b = np.histogram(arr, bins=bins)
    x = (b[:-1] + b[1:]) / 2
    
    return n/n.max(), b, x



def plot_highest_score(combos):
    score_mask = combos.high_score_combo_mask
    high_scores = combos.evt_highest_score
    signal_mask = combos.signal_evt_mask
    signal_highs = combos.signal_high_score
    high_nsignal = combos.highest_score_nsignal
    n_signal = combos.n_signal
    
    fig, ax = plt.subplots()
    score_bins = np.arange(0, 1.01, 0.01)

    n_signal, edges = np.histogram(high_scores[signal_mask], bins=score_bins)
    n_bkgd, edges   = np.histogram(high_scores[~signal_mask], bins=score_bins)

    x = (edges[1:] + edges[:-1])/2

    n_signal = n_signal / np.sum(n_signal)
    n_bkgd = n_bkgd / np.sum(n_bkgd)

    n_signal, edges, _ = hist(ax, x, weights=n_signal, bins=score_bins, label='Events with correct combos')
    n_bkgd, edges, _ = hist(ax, x, weights=n_bkgd, bins=score_bins, label='Events with no correct combos')

    ax.legend(loc=2)
    ax.set_xlabel('Highest Assigned Score in Event')
    ax.set_ylabel('AU')
    ax.set_title('Distribution of Highest Scoring Combination')

    # hi_score = np.sum(n_signal[x > 0.8]) / (np.sum(n_signal[x > 0.8]) + np.sum(n_bkgd[x > 0.8]))
    # ax.text(0.2, 0.5, f"Ratio of signal to sgnl+bkgd above 0.8 = {hi_score*100:.0f}%", transform=ax.transAxes)

    return fig, ax



def plot_combo_scores(combos, normalize=True):

    fig, ax = plt.subplots()

    if normalize:
        c, b, x = norm_hist(combos.scores_combo[combos.signal_mask])
        w, b, x = norm_hist(combos.scores_combo[~combos.signal_mask])
        ax.set_ylabel('AU')
    else:
        c, b = np.histogram(combos.scores_combo[combos.signal_mask], bins=100)
        w, b = np.histogram(combos.scores_combo[~combos.signal_mask], bins=100)
        x = (b[1:] + b[:-1]) / 2
        ax.set_ylabel('Entries Per Bin')

    hist(ax, x, weights=c, bins=b, label='Correct 6-jet combo')
    hist(ax, x, weights=w, bins=b, label='Incorrect 6-jet combo')
    ax.legend(fontsize='small', loc=9)

    ax.set_xlabel('Assigned Score')
    

    textstr = f'Entries = {len(combos.scores_combo)}'
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax.text(0.8, 1.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    return fig, ax

def plot_combo_score_v_mass(combos):
    combo_m = ak.to_numpy(combos.sixjet_p4.mass)
    combo_m = combo_m.reshape(combo_m.shape[0])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))

    fig.suptitle("Combination Analysis")

    ax[0].set_title("Correct Combos")
    ax[1].set_title("Incorrect Combos")

    n, xedges, yedges, ims = hist2d(ax[0], combo_m[combos.signal_mask], combos.scores_combo[combos.signal_mask], xbins=np.linspace(400,900,100))
    n, xedges, yedges, imb = hist2d(ax[1], combo_m[~combos.signal_mask], combos.scores_combo[~combos.signal_mask], xbins=np.linspace(0,2000,100))

    plt.colorbar(ims, ax=ax[0])
    plt.colorbar(imb, ax=ax[1])

    ax[0].set_xlabel('Invariant Mass of 6-jet System [GeV]')
    ax[1].set_xlabel('Invariant Mass of 6-jet System [GeV]')
    ax[0].set_ylabel('Assigned Score')
    ax[1].set_ylabel('Assigned Score')

    plt.tight_layout()
    return fig, ax

def plot_highest_score_v_mass(combos):
    combo_m = ak.to_numpy(combos.sixjet_p4.mass)
    combo_m = combo_m.reshape(combo_m.shape[0])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))

    fig.suptitle("Combination Analysis")

    ax[0].set_title("Correct Combos")
    ax[1].set_title("Incorrect Combos")
    
    signal_mask = combos.signal_evt_mask
    high_score_mask = combos.high_score_combo_mask

    n, xedges, yedges, ims = hist2d(ax[0], combo_m[high_score_mask][signal_mask], combos.scores_combo[high_score_mask][signal_mask], xbins=np.linspace(400,900,100))
    n, xedges, yedges, imb = hist2d(ax[1], combo_m[high_score_mask][~signal_mask], combos.scores_combo[high_score_mask][~signal_mask], xbins=np.linspace(0,2000,100))

    plt.colorbar(ims, ax=ax[0])
    plt.colorbar(imb, ax=ax[1])

    ax[0].set_xlabel('Invariant Mass of 6-jet System [GeV]')
    ax[1].set_xlabel('Invariant Mass of 6-jet System [GeV]')
    ax[0].set_ylabel('Assigned Score')
    ax[1].set_ylabel('Assigned Score')

    plt.tight_layout()
    return fig, ax


plot_dict = {
    'histtype' : 'step',
    'align' : 'mid',
    'linewidth' : 2
    }
def plot(x, scale=False, **kwargs):
    """
    This function is a wrapper for matplotlib.pyplot.hist that allows me to generate histograms quickly and consistently.
    It also helps deal with background trees, which are typically given as lists of samples.
    """
    if 'ax' not in kwargs: fig, ax = plt.subplots(figsize=(10,6))
    else: 
        fig, ax = kwargs['fig'], kwargs['ax']
        kwargs.pop('fig')
        kwargs.pop('ax')
    for k,v in plot_dict.items():
        if k not in kwargs: kwargs[k] = v
    if type(x) == list:
        bins = kwargs['bins']
        n = np.zeros_like()
        for bkg_kin, scale in zip(x, scale):
            n_temp, e = np.histogram(bkg_kin.to_numpy(), bins=bins)
            n += n_temp*scale
        n, edges, im = ax.hist(x=(bins[1:] + bins[:-1])/2, weights=n, **kwargs)
    else:
        if 'density' in kwargs.keys():
            if kwargs['density'] == 1:
                kwargs.pop('density')
                bins = kwargs['bins']
                try:
                    n, e = np.histogram(x, bins)
                except:
                    n, e = np.histogram(x.to_numpy(), bins)
                n, edges, im = ax.hist(x=(bins[1:] + bins[:-1])/2, weights=n/n.sum(), **kwargs)
        else:
            n, edges, im = ax.hist(x, **kwargs)
    if 'label' in kwargs: ax.legend()
    return fig, ax, n, edges

def plot2d(x, y, **kwargs):
    return plt.hist2d(x, y, **kwargs)