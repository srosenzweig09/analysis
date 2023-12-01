from utils.analysis import Bkg
from utils.files import *
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import awkward as ak
from utils.plotter import Hist
import matplotlib.pyplot as plt

from rich import print as rprint
from rich.console import Console
console = Console()

enriched = Bkg(get_qcd_enriched('cutflow_studies/nocuts'))
genfilter = Bkg(get_qcd_bgen('cutflow_studies/nocuts'))
# htbins = Bkg(get_qcd_pt('cutflow_studies/nocuts'))

# htbin_strings = []
# for sample in enriched.sample:
#     htbin_strings.append([out for out in sample.split('_') if 'HT' in out][0])

# indices = []
# for bins in enriched:
#     for i,sample in enumerate(enriched.sample):
#         if bins in sample:
#             indices.append(int(i))
#             break

# jet and b jet multiplicity
# pt of jet and b
# HT
# MX, MY
# number of gen b quarks and b tagged jets

# enriched_scale = np.repeat(enriched.scale, ak.concatenate([ak.count(pt, axis=1) for pt in enriched.jet_pt]).to_numpy()) 
# enriched_norm_scale = enriched_scale / enriched_scale.sum()
# genfilter_scale = np.repeat(genfilter.scale, ak.concatenate([ak.count(pt, axis=1) for pt in genfilter.jet_pt]).to_numpy())
# genfilter_norm_scale = genfilter_scale / genfilter_scale.sum()
# htbins_scale = np.repeat(htbins.scale, ak.concatenate([ak.count(pt, axis=1) for pt in htbins.jet_pt]).to_numpy())
# htbins_norm_scale = htbins_scale / htbins_scale.sum()

enriched_b_mask = [ak.sum(abs(x) == 5, axis=1) for x in enriched.jet_partonFlav]
genfilter_b_mask = [ak.sum(abs(x) == 5, axis=1) for x in genfilter.jet_partonFlav]
# htbins_b_mask = [ak.sum(abs(x) == 5, axis=1) for x in htbins.jet_partonFlav]

pdfname = 'plots/qcd/qcd_comparison.pdf'
console.log(f"Writing to {pdfname}")
with PdfPages(pdfname) as pdf:
    ###############################################
    # Jet multiplicity
    ###############################################

    console.log("Plotting jet multiplicity")

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(40,40))

    bins = np.arange(21)
    for i,ax in enumerate(axs.flatten()[:-1]):
        # j = indices[i]

        Hist(enriched.get('n_jet')[i], bins=bins, ax=ax, label=enriched.sample[i], density=True)
        Hist(genfilter.get('n_jet')[i], bins=bins, ax=ax, label=genfilter.sample[i], density=True)
        # Hist(htbins.get('n_jet')[j], bins=bins, ax=ax, label=htbins.sample[j], density=True)
        
        ax.set_xticks(range(0,21,2))
        ax.set_xlabel(r"Number of Jets in Event")
        ax.set_ylabel('AU')
        ax.legend()

    fig.suptitle('Reco Jet Multiplicity', y=0.9, fontsize=40)

    console.log("Writing jet multiplicity plot to pdf")

    pdf.savefig()
    plt.close()

    console.log("Plotting reco jet multiplicity")

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(40,40))

    bins = np.arange(7)
    for i,ax in enumerate(axs.flatten()[:-1]):
        # j = indices[i]

        Hist(ak.sum(abs(enriched.jet_partonFlav[i]) == 5, axis=1), bins=bins, ax=ax, label=enriched.sample[i], density=True)
        Hist(ak.sum(abs(genfilter.jet_partonFlav[i]) == 5, axis=1), bins=bins, ax=ax, label=genfilter.sample[i], density=True)
        # Hist(ak.sum(abs(htbins.jet_partonFlav[j]) == 5, axis=1), bins=bins, ax=ax, label=htbins.sample[j], density=True)
        
        ax.set_xticks(range(7))
        ax.set_xlabel(r"Number of Reco b Jets in Event")
        ax.set_ylabel('AU')
        ax.legend()

    fig.suptitle('Reco b Jet Multiplicity', y=0.9, fontsize=40)

    console.log("Writing reco jet multiplicity plot to pdf")

    pdf.savefig()
    plt.close()

    console.log("Plotting gen jet multiplicity")

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(40,40))

    bins = np.arange(7)
    for i,ax in enumerate(axs.flatten()[:-1]):
        # j = indices[i]

        Hist(ak.sum(abs(enriched.genjet_partonFlav[i]) == 5, axis=1), bins=bins, ax=ax, label=enriched.sample[i], density=True)
        Hist(ak.sum(abs(genfilter.genjet_partonFlav[i]) == 5, axis=1), bins=bins, ax=ax, label=genfilter.sample[i], density=True)
        # Hist(ak.sum(abs(htbins.genjet_partonFlav[j]) == 5, axis=1), bins=bins, ax=ax, label=htbins.sample[j], density=True)
        
        ax.set_xticks(range(7))
        ax.set_xlabel(r"Number of Gen b Jets in Event")
        ax.set_ylabel('AU')
        ax.legend()

    fig.suptitle('Gen b Jet Multiplicity', y=0.9, fontsize=40)

    console.log("Writing gen jet multiplicity plot to pdf")

    pdf.savefig()
    plt.close()


    ###############################################
    # b-tagged jet multiplicity
    ###############################################

    enriched_n_loose_bs = [ak.sum(x > 0.0490, axis=1) for x in enriched.get('jet_btag')]
    enriched_n_medium_bs = [ak.sum(x > 0.2783, axis=1) for x in enriched.get('jet_btag')]
    enriched_n_tight_bs = [ak.sum(x > 0.7100, axis=1) for x in enriched.get('jet_btag')]

    genfilter_n_loose_bs = [ak.sum(x > 0.0490, axis=1) for x in genfilter.get('jet_btag')]
    genfilter_n_medium_bs = [ak.sum(x > 0.2783, axis=1) for x in genfilter.get('jet_btag')]
    genfilter_n_tight_bs = [ak.sum(x > 0.7100, axis=1) for x in genfilter.get('jet_btag')]

    # htbins_n_loose_bs = [ak.sum(x > 0.0490, axis=1) for x in htbins.get('jet_btag')]
    # htbins_n_medium_bs = [ak.sum(x > 0.2783, axis=1) for x in htbins.get('jet_btag')]
    # htbins_n_tight_bs = [ak.sum(x > 0.7100, axis=1) for x in htbins.get('jet_btag')]

    console.log("Plotting loose b-tagged jet multiplicity")

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(40,40))

    bins = np.arange(20)
    for i,ax in enumerate(axs.flatten()[:-1]):
        # j = indices[i]

        Hist(enriched_n_loose_bs[i], bins=bins, ax=ax, label=enriched.sample[i], density=True)
        Hist(genfilter_n_loose_bs[i], bins=bins, ax=ax, label=genfilter.sample[i], density=True)
        # Hist(htbins_n_loose_bs[j], bins=bins, ax=ax, label=htbins.sample[j], density=True)
        
        ax.set_xticks(range(0,21,2))
        ax.set_xlabel(r"Number of Reco Loose b-Tagged Jets in Event")
        ax.set_ylabel('AU')
        ax.legend()

    fig.suptitle('Loose b Tags', y=0.9, fontsize=40)

    console.log("Writing loose b-tagged jet multiplicity plot to pdf")

    pdf.savefig()
    plt.close()

    console.log("Plotting medium b-tagged jet multiplicity")

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(40,40))

    bins = np.arange(20)
    for i,ax in enumerate(axs.flatten()[:-1]):
        # j = indices[i]

        Hist(enriched_n_medium_bs[i], bins=bins, ax=ax, label=enriched.sample[i], density=True)
        Hist(genfilter_n_medium_bs[i], bins=bins, ax=ax, label=genfilter.sample[i], density=True)
        # Hist(htbins_n_medium_bs[j], bins=bins, ax=ax, label=htbins.sample[j], density=True)
        
        ax.set_xticks(range(0,21,2))
        ax.set_xlabel(r"Number of Reco Medium b-Tagged Jets in Event")
        ax.set_ylabel('AU')
        ax.legend()

    fig.suptitle('Medium b Tags', y=0.9, fontsize=40)

    console.log("Writing medium b-tagged jet multiplicity plot to pdf")

    pdf.savefig()
    plt.close()

    console.log("Plotting tight b-tagged jet multiplicity")

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(40,40))

    bins = np.arange(20)
    for i,ax in enumerate(axs.flatten()[:-1]):
        # j = indices[i]

        Hist(enriched_n_tight_bs[i], bins=bins, ax=ax, label=enriched.sample[i], density=True)
        Hist(genfilter_n_tight_bs[i], bins=bins, ax=ax, label=genfilter.sample[i], density=True)
        # Hist(htbins_n_tight_bs[j], bins=bins, ax=ax, label=htbins.sample[j], density=True)
        
        ax.set_xticks(range(0,21,2))
        ax.set_xlabel(r"Number of Reco Tight b-Tagged Jets in Event")
        ax.set_ylabel('AU')
        ax.legend()

    fig.suptitle('Tight b Tags', y=0.9, fontsize=40)

    console.log("Writing tight b-tagged jet multiplicity plot to pdf")

    pdf.savefig()
    plt.close()

    ###############################################
    # Jet pt
    ###############################################

    console.log("Plotting jet pt")

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(40,40))

    bins = np.linspace(0,100,41)
    for i,ax in enumerate(axs.flatten()[:-1]):
        # j = indices[i]

        Hist(enriched.jet_pt[i], bins=bins, ax=ax, label=enriched.sample[i], density=True)
        Hist(genfilter.jet_pt[i], bins=bins, ax=ax, label=genfilter.sample[i], density=True)
        # Hist(htbins.jet_pt[j], bins=bins, ax=ax, label=htbins.sample[j], density=True)
        
        # ax.set_xticks(range(0,21,2))
        ax.set_xlabel(r"$p_T$ [GeV]")
        ax.set_ylabel('AU')
        ax.legend()
    
    fig.suptitle(r'Reco Jet $p_T$', y=0.9, fontsize=40)

    console.log("Writing jet pt plot to pdf")

    pdf.savefig()
    plt.close()

    console.log("Plotting jet pt for jets matched to b quarks")

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(40,40))

    bins = np.linspace(0,100,41)
    for i,ax in enumerate(axs.flatten()[:-1]):
        # j = indices[i]

        Hist(enriched.jet_pt[i][enriched_b_mask[i]], bins=bins, ax=ax, label=enriched.sample[i], density=True)
        Hist(genfilter.jet_pt[i][genfilter_b_mask[i]], bins=bins, ax=ax, label=genfilter.sample[i], density=True)
        # Hist(htbins.jet_pt[j][htbins_b_mask[j]], bins=bins, ax=ax, label=htbins.sample[j], density=True)
        
        # ax.set_xticks(range(0,21,2))
        ax.set_xlabel(r"$p_T$ [GeV]")
        ax.set_ylabel('AU')
        ax.legend()
    
    fig.suptitle(r'Reco b Jet $p_T$', y=0.9, fontsize=40)

    console.log("Writing jet pt plot to pdf")

    pdf.savefig()
    plt.close()


    ###############################################
    # HT
    ###############################################

    L = [0, 0, 0, 200, 500, 700, 1000, 1000]
    H = [500, 700, 1000, 1100, 1500, 2000, 2500, 4500]

    console.log("Plotting HT")

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(40,40))

    for i,ax in enumerate(axs.flatten()[:-1]):
        # j = indices[i]

        l,h = L[i],H[i]
        bins = np.linspace(l,h,41)

        Hist(ak.sum(enriched.jet_pt[i], axis=1), bins=bins, ax=ax, label=enriched.sample[i], density=True)
        Hist(ak.sum(genfilter.jet_pt[i], axis=1), bins=bins, ax=ax, label=genfilter.sample[i], density=True)
        # Hist(ak.sum(htbins.jet_pt[j], axis=1), bins=bins, ax=ax, label=htbins.sample[j], density=True)
        
        ax.set_xlabel(r"$H_T$ [GeV]")
        ax.set_ylabel('AU')
        ax.legend()
    
    fig.suptitle(r'Reco Jet $H_T$', y=0.9, fontsize=40)

    console.log("Writing HT plot to pdf")

    pdf.savefig()
    plt.close()

    console.log("Plotting gen jet HT")

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(40,40))

    for i,ax in enumerate(axs.flatten()[:-1]):
        # j = indices[i]

        l,h = L[i],H[i]
        bins = np.linspace(l,h,41)

        Hist(ak.sum(enriched.genjet_pt[i], axis=1), bins=bins, ax=ax, label=enriched.sample[i], density=True)
        Hist(ak.sum(genfilter.genjet_pt[i], axis=1), bins=bins, ax=ax, label=genfilter.sample[i], density=True)
        # Hist(ak.sum(htbins.jet_pt[j], axis=1), bins=bins, ax=ax, label=htbins.sample[j], density=True)
        
        ax.set_xlabel(r"$H_T$ [GeV]")
        ax.set_ylabel('AU')
        ax.legend()
    
    fig.suptitle(r'Gen Jet $H_T$', y=0.9, fontsize=40)

    console.log("Writing HT plot to pdf")

    pdf.savefig()
    plt.close()

    console.log("Plotting HT for jets matched to b quarks")

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(40,40))

    for i,ax in enumerate(axs.flatten()[:-1]):
        # j = indices[i]

        l,h = L[i],H[i]
        bins = np.linspace(l,h,41)

        Hist(ak.sum(enriched.jet_pt[i][enriched_b_mask[i]], axis=1), bins=bins, ax=ax, label=enriched.sample[i], density=True)
        Hist(ak.sum(genfilter.jet_pt[i][genfilter_b_mask[i]], axis=1), bins=bins, ax=ax, label=genfilter.sample[i], density=True)
        # Hist(ak.sum(htbins.jet_pt[j][htbins_b_mask[j]], axis=1), bins=bins, ax=ax, label=htbins.sample[j], density=True)
        
        ax.set_xlabel(r"$H_T$ [GeV]")
        ax.set_ylabel('AU')
        ax.legend()
    
    fig.suptitle(r'Reco b Jet $H_T$', y=0.9, fontsize=40)

    console.log("Writing HT plot to pdf")

    pdf.savefig()
    plt.close()

    ###############################################
    # MX, MY
    ###############################################

    console.log("Plotting MX, MY")

    L = [0]*8
    H = [4000, 4000, 4000, 4250, 4500, 5000, 6000, 7000]
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(40,40))

    for i,ax in enumerate(axs.flatten()[:-1]):
        # j = indices[i]

        l,h = L[i],H[i]
        bins = np.linspace(l,h,41)

        n = Hist(enriched.X_m[i], bins=bins, ax=ax, label=enriched.sample[i], density=True)
        n = Hist(genfilter.X_m[i], bins=bins, ax=ax, label=genfilter.sample[i], density=True)
        # n = Hist(htbins.X_m[j], bins=bins, ax=ax, label=htbins.sample[j], density=True)

        ax.set_xlabel(r"$M_X$ [GeV]")
        ax.set_ylabel("AU")
        ax.legend()

    fig.suptitle(r'Reco $M_X$', y=0.9, fontsize=40)

    console.log("Writing MX plot to pdf")

    pdf.savefig()
    plt.close()

    L = [0]*8
    H = [2500, 2500, 2500, 3000, 3000, 3000, 4000, 4500]

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(40,40))

    bins = np.linspace(0,10000,41)
    for i,ax in enumerate(axs.flatten()[:-1]):
        # j = indices[i]

        l,h = L[i],H[i]
        bins = np.linspace(l,h,41)

        n = Hist(enriched.Y_m[i], bins=bins, ax=ax, label=enriched.sample[i], density=True)
        n = Hist(genfilter.Y_m[i], bins=bins, ax=ax, label=genfilter.sample[i], density=True)
        # n = Hist(htbins.Y_m[j], bins=bins, ax=ax, label=htbins.sample[j], density=True)

        ax.set_xlabel(r"$M_Y$ [GeV]")
        ax.set_ylabel("AU")
        ax.legend()

    fig.suptitle(r'Reco $M_Y$', y=0.9, fontsize=40)

    console.log("Writing MY plot to pdf")

    pdf.savefig()
    plt.close()

print("Done!")
print("Saved to:", pdfname)