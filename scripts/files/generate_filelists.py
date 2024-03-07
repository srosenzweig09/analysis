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
def generate_central_filelist(year):
    # base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/*/ntuple.root'
    base = f'/cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/{year}/maxbtag_4b/Official_NMSSM/*/ntuple.root'
    files = glob.glob(base)
    files = [f"{out}\n" for out in files if get_central_mx(out) < 1300]
    files = sorted(files)
    files[-1] = files[-1].replace('\n', '')
    with open(f"filelists/{year}/central.txt", "w") as f:
        f.writelines(files)

# # Private Samples ######################################
def generate_private_filelist(year):
    # base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM/*/ntuple.root'
    base = f'/cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/{year}/maxbtag/NMSSM/*/ntuple.root'
    files = glob.glob(base)
    files = [f"{out}\n" for out in files if get_private_mx(out) < 1300]
    files = sorted(files)
    files[-1] = files[-1].replace('\n', '')
    with open(f"filelists/private.txt", "w") as f:
        f.writelines(files)

# Systematic Samples ######################################
def generate_central_systematics_filelist(year):
    # base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/syst/*/*/*/ntuple.root'
    base = f'/cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/{year}/maxbtag_4b/Official_NMSSM/syst/*/*/*/ntuple.root'
    files = glob.glob(base)
    files = [f"{out}\n" for out in files if 'analysis_tar' not in out]
    files = sorted(files)
    files[-1] = files[-1].replace('\n', '')
    with open(f"filelists/{year}/central_systematics.txt", "w") as f:
        f.writelines(files)

def generate_background_filelist(year):
    # base = f'/eos/uscms/store/user/srosenzw/sixb/ntuples/{year}/maxbtag_4b'
    base = f'/cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/{year}/maxbtag_4b'
    files = glob.glob(f"{base}/QCD/*") + glob.glob(f"{base}/TTJets/*")
    files = sorted([f"{f}/ntuple.root\n" for f in files if 'analysis_tar' not in f and 'small_' not in f])
    files[-1] = files[-1].replace('\n', '')
    with open(f"filelists/{year}/background.txt", "w") as f:
        f.writelines(files)

def generate_data_filelist(year):
    # base = f'/eos/uscms/store/user/srosenzw/sixb/ntuples/{year}/maxbtag_4b/{year_dict[year]}/ntuple.root'
    base = f'/cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/{year}/maxbtag_4b/{year_dict[year]}/ntuple.root'
    files = glob.glob(base)
    with open(f"filelists/{year}/data.txt", "w") as f:
        f.writelines(files)


year = 'Summer2018UL'
year_dict = {
    'Summer2018UL' : 'JetHT_Data_UL'
}

# generate_central_filelist(year)
generate_private_filelist(year)
# generate_central_systematics_filelist(year)
# generate_background_filelist(year)
# generate_data_filelist(year)

