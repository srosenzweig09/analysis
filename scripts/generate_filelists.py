"""
Writes txt files containing the root file locations for the nominal and systematic central signal samples, as well as the privately-produced samples used to calculate the correction ratio for the b tag sf.
"""
import re, glob
import sys

def get_central_mx(out):
    return int(out.split('-')[1].split('_')[0])

def get_private_mx(out):
    return int(re.search('MX_(.*)_MY_', out).group(1))

# Central Samples ######################################
def get_central_samples():
    # base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/*/ntuple.root'
    base = '/cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/*/ntuple.root'
    files = glob.glob(base)
    masses = [f"{out}\n" for out in files if get_central_mx(out) < 1300]
    with open("filelists/central.txt", "w") as f:
        f.writelines(masses)

# masses = [f"root://cmseos.fnal.gov//store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/{out}/ntuple.root\n" for out in output if get_central_mx(out) < 1300] # with spaces for FeynNet prediction
# with open("filelists/central_feynman.txt", "w") as f:
#     f.writelines(masses)

# # Private Samples ######################################
def get_private_samples():
    # base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM/*/ntuple.root'
    base = '/cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM/*/ntuple.root'
    output = glob.glob(base)
    output = [f"{out}\n" for out in output if get_private_mx(out) < 1300]
    with open("filelists/private.txt", "w") as f:
        f.writelines(output)

# Systematic Samples ######################################
def get_central_systematics():
    # base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/syst/*/*/*/ntuple.root'
    base = '/cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/syst/*/*/*/ntuple.root'
    files = glob.glob(base)
    files = [f"{out}\n" for out in files]
    with open("filelists/central_systematics.txt", "w") as f:
        f.writelines(files)

get_central_samples()
get_private_samples()
get_central_systematics()