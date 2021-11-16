from utils.analysis import Tree
from utils.plotter import Hist
from utils.varUtils import *
from utils.fileUtils.sr import NMSSM_List, JetHT_Data_UL
from utils.fileUtils.misc import DataCards_dir, buildDataCard

import awkward as ak
from colorama import Fore, Style
import vector

def getKey(tree, key):
    return tree.get('t6_jet_' + key)

def Print(line):
    if isinstance(line, str): print(Fore.CYAN + line + Style.RESET_ALL)
    else: print(Fore.CYAN + str(line) + Style.RESET_ALL)

# Threshold Parameters
cut_6jNN = 0 # NN6j threshold
mass_veto = 60 # Higgs mass hypothesis window (half-length)

Print("Processing data...")

# Data Tree Handling
datTree = Tree(JetHT_Data_UL)
datHiggsId = datTree.t6_jet_higgsIdx

H1_b1 = vector.obj(
    pt=getKey(datTree,'pt')[datHiggsId == 0][:,0], 
    eta=getKey(datTree,'eta')[datHiggsId == 0][:,0], 
    phi=getKey(datTree,'phi')[datHiggsId == 0][:,0], 
    m=getKey(datTree,'m')[datHiggsId == 0][:,0])
H1_b2 = vector.obj(
    pt=getKey(datTree,'pt')[datHiggsId == 0][:,1], 
    eta=getKey(datTree,'eta')[datHiggsId == 0][:,1], 
    phi=getKey(datTree,'phi')[datHiggsId == 0][:,1], 
    m=getKey(datTree,'m')[datHiggsId == 0][:,1])
H2_b1 = vector.obj(
    pt=getKey(datTree,'pt')[datHiggsId == 1][:,0], 
    eta=getKey(datTree,'eta')[datHiggsId == 1][:,0], 
    phi=getKey(datTree,'phi')[datHiggsId == 1][:,0], 
    m=getKey(datTree,'m')[datHiggsId == 1][:,0])
H2_b2 = vector.obj(
    pt=getKey(datTree,'pt')[datHiggsId == 1][:,1], 
    eta=getKey(datTree,'eta')[datHiggsId == 1][:,1], 
    phi=getKey(datTree,'phi')[datHiggsId == 1][:,1], 
    m=getKey(datTree,'m')[datHiggsId == 1][:,1])
H3_b1 = vector.obj(
    pt=getKey(datTree,'pt')[datHiggsId == 2][:,0], 
    eta=getKey(datTree,'eta')[datHiggsId == 2][:,0], 
    phi=getKey(datTree,'phi')[datHiggsId == 2][:,0], 
    m=getKey(datTree,'m')[datHiggsId == 2][:,0])
H3_b2 = vector.obj(
    pt=getKey(datTree,'pt')[datHiggsId == 2][:,1], 
    eta=getKey(datTree,'eta')[datHiggsId == 2][:,1], 
    phi=getKey(datTree,'phi')[datHiggsId == 2][:,1], 
    m=getKey(datTree,'m')[datHiggsId == 2][:,1])

X = H1_b1 + H1_b2 + H2_b1 + H2_b2 + H3_b1 + H3_b2

# apply NN6j selection
data_6jNN_mask = datTree.b_6j_score > cut_6jNN # pass 6jNN mask

# mass veto selection
data_CR_mask = abs(datTree.t6_higgs_m[:,2] - 125) > mass_veto # CR
data_CR = data_6jNN_mask & data_CR_mask

data_sums = ak.sum(datTree.t6_jet_btag, axis=1)[data_CR]/6
n_data, edges = np.histogram(data_sums.to_numpy(), bins=score_bins)

Print("Processing signal...")
for MASS_PAIR in NMSSM_List:
    mxmy = MASS_PAIR.split('/')[-2]
    Print(mxmy)
    sigTree = Tree(MASS_PAIR)
    sigHiggsId = sigTree.t6_jet_higgsIdx

    sig_H1_b1 = vector.obj(
        pt=getKey(sigTree,'pt')[sigHiggsId == 0][:,0], 
        eta=getKey(sigTree,'eta')[sigHiggsId == 0][:,0], 
        phi=getKey(sigTree,'phi')[sigHiggsId == 0][:,0], 
        m=getKey(sigTree,'m')[sigHiggsId == 0][:,0])
    sig_H1_b2 = vector.obj(
        pt=getKey(sigTree,'pt')[sigHiggsId == 0][:,1], 
        eta=getKey(sigTree,'eta')[sigHiggsId == 0][:,1], 
        phi=getKey(sigTree,'phi')[sigHiggsId == 0][:,1], 
        m=getKey(sigTree,'m')[sigHiggsId == 0][:,1])
    sig_H2_b1 = vector.obj(
        pt=getKey(sigTree,'pt')[sigHiggsId == 1][:,0], 
        eta=getKey(sigTree,'eta')[sigHiggsId == 1][:,0], 
        phi=getKey(sigTree,'phi')[sigHiggsId == 1][:,0], 
        m=getKey(sigTree,'m')[sigHiggsId == 1][:,0])
    sig_H2_b2 = vector.obj(
        pt=getKey(sigTree,'pt')[sigHiggsId == 1][:,1], 
        eta=getKey(sigTree,'eta')[sigHiggsId == 1][:,1], 
        phi=getKey(sigTree,'phi')[sigHiggsId == 1][:,1], 
        m=getKey(sigTree,'m')[sigHiggsId == 1][:,1])
    sig_H3_b1 = vector.obj(
        pt=getKey(sigTree,'pt')[sigHiggsId == 2][:,0], 
        eta=getKey(sigTree,'eta')[sigHiggsId == 2][:,0], 
        phi=getKey(sigTree,'phi')[sigHiggsId == 2][:,0], 
        m=getKey(sigTree,'m')[sigHiggsId == 2][:,0])
    sig_H3_b2 = vector.obj(
        pt=getKey(sigTree,'pt')[sigHiggsId == 2][:,1], 
        eta=getKey(sigTree,'eta')[sigHiggsId == 2][:,1], 
        phi=getKey(sigTree,'phi')[sigHiggsId == 2][:,1], 
        m=getKey(sigTree,'m')[sigHiggsId == 2][:,1])

    sig_X = sig_H1_b1 + sig_H1_b2 + sig_H2_b1 + sig_H2_b2 + sig_H3_b1 + sig_H3_b2


    sgnl_6jNN_mask = sigTree.b_6j_score > cut_6jNN # pass 6jNN mask

    # mass veto
    sgnl_CR_mask = abs(sigTree.t6_higgs_m[:,2] - 125) > mass_veto # CR
    sgnl_SR_mask = ~sgnl_CR_mask # SR
    sgnl_CR = sgnl_6jNN_mask & sgnl_CR_mask
    sgnl_SR = sgnl_6jNN_mask & sgnl_SR_mask

    sig_sums = ak.sum(sigTree.t6_jet_btag, axis=1)[sgnl_SR_mask]/6
    n_sig, edges = np.histogram(sig_sums.to_numpy(), bins=score_bins)

    sum6_eff = []
    sum6_rej = []

    for cut in edges[:-1]:
        sum6_eff.append(n_sig[edges[:-1] >= cut].sum()/n_sig.sum())
        sum6_rej.append(n_data[edges[:-1] < cut].sum()/n_data.sum())

    sum6_eff = np.append(1, np.asarray(sum6_eff))
    sum6_rej = np.asarray(sum6_rej)

    dx = sum6_eff[:-1] - sum6_eff[1:]
    auc = np.sum(sum6_rej*dx)
    sum6_rej = np.append(sum6_rej, 1)

    opt_arg = (abs(sum6_eff-auc)+abs(sum6_rej-auc)).argmin()

    # opt_cut = True
    opt_cut = False
    if opt_cut:
        opt_cut = score_bins[opt_arg]
        Print(Fore.GREEN + f"Optimal score cut = {opt_cut}")
    else:
        opt_cut = 0.64
        Print(Fore.GREEN + f"PRESET score cut = {opt_cut}")

    ### SIGNAL ###
    # score veto
    sgnl_fail_btag_mask = ak.sum(sigTree.t6_jet_btag, axis=1)/6 < opt_cut # ls
    sgnl_pass_btag_mask = ak.sum(sigTree.t6_jet_btag, axis=1)/6 >= opt_cut # hs

    # combination
    sgnl_SRhs_mask = sgnl_SR_mask & sgnl_pass_btag_mask
    sgnl_SRls_mask = sgnl_SR_mask & sgnl_fail_btag_mask

    #### DATA ####
    # score veto
    data_ls_mask = ak.sum(datTree.t6_jet_btag, axis=1)/6 < opt_cut # ls
    data_hs_mask = ak.sum(datTree.t6_jet_btag, axis=1)/6 >= opt_cut # hs

    # combination
    data_CRls_mask = data_6jNN_mask & data_CR_mask & data_ls_mask
    data_CRhs_mask = data_6jNN_mask & data_CR_mask & data_hs_mask
    data_SRls_mask = data_6jNN_mask & ~data_CR_mask & data_ls_mask

    TF = ak.sum(data_CRhs_mask)/ak.sum(data_CRls_mask)
    Print(f"{TF:.3f}")

    max_mass = 2000

    mX = int(sigTree.sample.split(' ')[1])
    mY = int(sigTree.sample.split(' ')[4])

    xmin = mX - 0.3*mX
    xmax = 800 + 0.15*mX
    nbins = 200

    ax, n, e = Hist(sig_X.m[sgnl_SRhs_mask], bins=np.linspace(0,max_mass,nbins), label='SR-hs', scale=sigTree.scale)
    S = int(n[(e[:-1]>xmin) & (e[:-1]<xmax)].sum())

    n, e = np.histogram(X.m[data_SRls_mask].to_numpy(), bins=np.linspace(0,max_mass,nbins))
    x = x_bins(e)
    n, e = Hist(x, weights=n*TF, bins=np.linspace(0,max_mass,nbins), ax=ax, label='bkg model')
    B = int(n[(e[:-1]>xmin) & (e[:-1]<xmax)].sum())

    sigma = int(np.sqrt(B))
    mu = 2*np.sqrt(B)/S
    sensitivity = sigTree.xsec*mu

    print(sigTree.mXmY)
    print(f"    Number of signal events = {S}")
    print(f"Number of background events = {B}")
    print(f"    Standard Deviation of B = {sigma}")
    print(f"                         mu = {mu:.3f}")
    print(Fore.LIGHTMAGENTA_EX + f"                      limit = {int(sensitivity*1000)} fb")
    print()

    # dcText = buildDataCard(S,B,.1)

    # dcName = f'datacard_{sigTree.mXmY}_sys.txt'
    # with open(DataCards_dir + dcName,"w") as f:
    #     f.writelines(dcText)
    # print("Created file")
    # print(dcName)