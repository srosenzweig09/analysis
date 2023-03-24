print("---------------------------------")
print("STARTING PROGRAM")
print("---------------------------------")

from argparse import ArgumentParser
from array import array
from configparser import ConfigParser
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import sys
from utils.analysis import Signal
from utils.plotter import Hist, Ratio

def getROOTCanvas_VR(h_title, bin_values):
   h1_title = h_title[0]
   h2_title = h_title[1]
   assert(h1_title == "h_obs")
   assert(h2_title == "h_est")
   h1_bin_vals = bin_values[0]
   h2_bin_vals = bin_values[1]
   print(f".. generating root hist for VR")
   canvas = ROOT.TCanvas('c1','c1', 600, 600)
   canvas.SetFrameLineWidth(3)

   h1 = ROOT.TH1D(h1_title,";m_{X} [GeV];Events",len(h1_bin_vals),array('d',list(mBins)))
   h2 = ROOT.TH1D(h2_title,";m_{X} [GeV];Events",len(h2_bin_vals),array('d',list(mBins)))

   for i,(vals1,vals2) in enumerate(zip(h1_bin_vals,h2_bin_vals)):
      h1.SetBinContent(i+1, vals1)
      h2.SetBinContent(i+1, vals2)

   print("ROOT KS",h1.KolmogorovTest(h2))

### ------------------------------------------------------------------------------------
## Implement command line parser

print(".. parsing command line arguments")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--cfg', dest='cfg', help='config file', required=True)

# region shapes
parser.add_argument('--rectangular', dest='rectangular', help='', action='store_true', default=True)
parser.add_argument('--spherical', dest='spherical', help='', action='store_true', default=False)
parser.add_argument('--CRedge', dest='CR', default=None)
parser.add_argument('--VRedge', dest='VR', default=None)
parser.add_argument('--SRedge', dest='SR', default=None)

args = parser.parse_args()

### ------------------------------------------------------------------------------------
## Implement config parser

print(".. parsing config file")

config = ConfigParser()
config.optionxform = str
config.read(args.cfg)

base = config['file']['base']
signal = config['file']['signal']
data = config['file']['data']
treename = config['file']['tree']
year = int(config['file']['year'])
pairing = config['pairing']['scheme']
pairing_type = pairing.split('_')[0]

minMX = int(config['plot']['minMX'])
maxMX = int(config['plot']['maxMX'])
nbins = int(config['plot']['nbins'])
mBins = np.linspace(minMX,maxMX,nbins)

if args.spherical and args.rectangular: args.rectangular = False
if args.rectangular: region_type = 'rect'
elif args.spherical: region_type = 'sphere'

score = float(config['score']['threshold'])

indir = f"root://cmseos.fnal.gov/{base}/{pairing}/"

if args.CR is not None: config['rectangular']['maxCR'] = args.CR
if args.VR is not None: config['rectangular']['maxVR'] = args.VR

### ------------------------------------------------------------------------------------
## Obtain data regions

datFileName = f"{indir}{data}"
datTree = Signal(datFileName)

if args.rectangular:
    print("\n ---RECTANGULAR---")
    datTree.rectangular_region(config)
elif args.spherical:
    print("\n---SPHERICAL---")
    datTree.spherical_region(config)
else:
    raise AttributeError("No mass region definition!")

dat_mX_CRls = datTree.dat_mX_CRls
dat_mX_CRhs = datTree.dat_mX_CRhs
dat_mX_VRls = datTree.dat_mX_VRls
dat_mX_VRhs = datTree.dat_mX_VRhs
dat_mX_SRls = datTree.dat_mX_SRls

# CR(data/model), VR (model), and SR (prediction)
VR_weights, SR_weights = datTree.bdt_process(region_type, config)
CR_weights = datTree.CR_weights

SR_edge, VR_edge, CR_edge = int(datTree.SR_edge), int(datTree.VR_edge), int(datTree.CR_edge)

x = (mBins[1:] + mBins[:-1])/2
region_dict = {
    'sr' : 'Signal Region',
    'vr' : 'Validation Region',
    'cr' : 'Control Region'
}
def ratioPlot(ls, weights, tag, hs=None):
    fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})
    ax = axs[0]
    ax.set_ylabel("Events")
    ax.set_title(f"{region_dict[tag]}, {SR_edge}, {VR_edge}, {CR_edge}")
    bin_mask = (ls > 375) & (ls < 2000)
    n_estimate, e = np.histogram(ls, weights=weights, bins=mBins)
    handles = [Rectangle([0,0],1,1,color='C0', fill=False, lw=2), Rectangle([0,0],1,1,color='C0', alpha=0.2)]
    labels=[f'Bkg Model ({int(n_estimate.sum())})', 'Bkg Uncertainty']
    ax = axs[1]
    ax.set_ylim(0.6,1.4)
    ax.set_ylabel('Bkg Unc')
    if hs is not None:
        n_target, _ = np.histogram(hs, bins=mBins)
        print(f"----> Percent Difference = {round(abs(n_target.sum()-n_estimate.sum())/n_target.sum()*100,1)}%")
        scatter = axs[0].scatter(x, n_target, color='k')
        if tag == 'cr': weights = n_target.sum() / sum(weights[bin_mask]) * weights
        n_estimate = Hist(ls, weights=weights, bins=mBins, ax=axs[0])
        n = n_target / n_estimate
        axs[1].scatter(x, n, color='k')
        axs[1].set_ylabel('Data/Model')
        handles.insert(0,scatter)
        labels.insert(0, f'Observed Data ({int(n_target.sum())})')
        for xval,n_dat,n_est,lbin in zip(x, n_target, n_estimate, mBins):
            width = xval - lbin
            rbin = xval + width
            dat_err = np.sqrt(n_dat)
            axs[0].plot([xval]*2, [n_dat-dat_err,n_dat+dat_err], color='k')
            if n_est == 0: n_est = 1
            dat_err_lo = (n_dat - dat_err) / n_est
            dat_err_hi = (n_dat + dat_err) / n_est
            axs[1].plot([xval]*2, [dat_err_lo,dat_err_hi], color='k')
    for xval,n_est,lbin in zip(x, n_estimate, mBins):
        width = xval - lbin
        rbin = xval + width
        est_mask = (ls > xval - width) & (ls < xval + width)
        est_err = np.sqrt(np.sum(weights[est_mask]**2))
        if n_est == 0: n_est = 1
        est_err_hi = (n_est + est_err) / n_est
        est_err_lo = (n_est - est_err) / n_est
        axs[0].fill_between([lbin, rbin], n_est-est_err, n_est+est_err, alpha=0.2, color='C0')
        axs[1].fill_between([lbin, rbin], est_err_lo, est_err_hi, color='C0', alpha=0.2)

    axs[0].legend(handles=handles, labels=labels)
    _ = ax.plot(mBins, np.ones_like(mBins), '--', color='gray')
    ax.set_xlabel(r"$m_X$ [GeV]")

    getROOTCanvas_VR(["h_obs", "h_est"], [n_target,n_estimate])

   #  fig.savefig(f"plots/data_{tag}_{SR_edge}_{VR_edge}_{CR_edge}.pdf", bbox_inches='tight')

print(".. plotting CR")
ratioPlot(hs=dat_mX_CRhs, ls=dat_mX_CRls, weights=CR_weights, tag='cr')

sys.exit()
print(".. plotting VR")
ratioPlot(hs=dat_mX_VRhs, ls=dat_mX_VRls, weights=VR_weights, tag='vr')
print(".. plotting SR")
ratioPlot(ls=dat_mX_SRls, weights=SR_weights, tag='sr')