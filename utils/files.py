import os, glob

hpg_base = '/cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples'

yearDict = {
   2018 : 'Summer2018UL',
   2017 : 'Summer2017UL',
   2016 : 'Summer2016UL',
   '2016preVFP' : 'Summer2016UL/preVFP'
}

maxbtag_4b_files = os.listdir(f"{hpg_base}/Summer2018UL/maxbtag_4b/Official_NMSSM")
maxbtag_files = os.listdir(f"{hpg_base}/Summer2018UL/maxbtag/NMSSM")

base = '/eos/uscms/store/user/srosenzw/sixb/ntuples'
data_path = {
   'Summer2018UL' : 'JetHT_Data_UL/ntuple.root',
   'Summer2017UL' : 'BTagCSV/ntuple.root',
}

def get_data(year='Summer2018UL', jets='btag_pt'):
   return f"{base}/{year}/{jets}/{data_path}"

def get_mpoint(flist, mx, my, private):
   if private: return [f for f in flist if f"_{mx}_" in f and f"_{my}" in f][0]
   else: return [f for f in flist if f"-{mx}_" in f and f"-{my}_" in f][0]

def get_NMSSM(mx=700, my=400, year=2018, selection='maxbtag_4b', suffix='', private=False):
   year = yearDict[year]
   flist = maxbtag_files if private else maxbtag_4b_files
   # mpoint = [f for f in flist if f"{mx}_" in f and f"{my}_" in f][0] # find filename with both mx and my in it
   mpoint = get_mpoint(flist, mx, my, private)
   if len(mpoint) < 0: raise FileNotFoundError(f'File not found: {mpoint}')
   if suffix != '': suffix = '_' + suffix
   mpoint = f"NMSSM/{mpoint}{suffix}" if private else f"Official_NMSSM/{mpoint}"
   return f"{base}/{year}/{selection}/{mpoint}/ntuple.root"

def get_NMSSM_list(year='Summer2018UL', jets='maxbtag_4b', private=False):
   location = f"{base}/{year}/{jets}/"
   if not private: location += "Official_"
   location += "NMSSM"
   flist = os.listdir(location)
   flist = [f"{location}/{f}/ntuple.root" for f in flist if 'NMSSM' in f]
   return flist

def get_nocuts(mx=700, my=400, year='Summer2018UL'):
   return f"{base}/{year}/cutflow_studies/nocuts/NMSSM/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}/ntuple.root"

def get_trigger(mx=700, my=400, year='Summer2018UL'):
   return f"{base}/{year}/cutflow_studies/trigger/NMSSM/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}/ntuple.root"

def get_presel(mx=700, my=400, year='Summer2018UL'):
   return f"{base}/{year}/cutflow_studies/presel/NMSSM/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}/ntuple.root"

def get_data(year='Summer2018UL', selection='maxbtag_4b'):
   return f"{base}/{year}/{selection}/{data_path[selection]}"

def get_qcd_benriched(selection, year='Summer2018UL'):
   files = sorted(glob.glob(f"{hpg_base}/{year}/{selection}/QCD/QCD_bEnriched_*/ntuple.root"))
   return files

def get_qcd_bgen(selection, year='Summer2018UL'):
   files = sorted(glob.glob(f"{hpg_base}/{year}/{selection}/QCD/QCD_*BGenFilter*/ntuple.root"))
   return files

def get_qcd_pt(selection, year='Summer2018UL'):
   files = sorted(glob.glob(f"{hpg_base}/{year}/{selection}/QCD/QCD_*PSWeights*/ntuple.root"))
   return files

def get_qcd_list(selection, year='Summer2018UL'):
   return get_qcd_benriched(selection, year) + get_qcd_bgen(selection, year)

def get_ttbar(selection, year='Summer2018UL', feyn=False):
   return f"{hpg_base}/{year}/{selection}/TTJets/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/ntuple.root"

def get_qcd_ttbar(selection, year='Summer2018UL', enriched=False, bgen=False, feyn=False):
   if enriched: return get_qcd_benriched(selection, year) + [get_ttbar(selection, year, feyn=feyn)]
   if bgen: return get_qcd_bgen(selection, year) + [get_ttbar(selection, year, feyn=feyn)]
   return get_qcd_list(selection, year) + [get_ttbar(selection, year, feyn=feyn)]
