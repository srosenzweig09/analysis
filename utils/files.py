import os

yearDict = {
   2018 : 'Summer2018UL',
   2017 : 'Summer2017UL',
   2016 : 'Summer2016UL',
   '2016preVFP' : 'Summer2016UL/preVFP'
}

maxbtag_4b_files = os.listdir("/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM")
maxbtag_files = os.listdir("/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM")

fnal_root = "root://cmseos.fnal.gov/"
base = '/eos/uscms/store/user/srosenzw/sixb/ntuples'
data_path = {
   'Summer2018UL' : 'JetHT_Data_UL/ntuple.root',
   'Summer2017UL' : 'BTagCSV/ntuple.root',
}
config = 'config/sphereConfig.cfg'

def get_data(run='Summer2018UL', jets='btag_pt'):
   return f"{base}/{run}/{jets}/{data_path}"

def get_NMSSM(mx=700, my=400, year=2018, selection='maxbtag_4b', suffix='', private=False):
   year = yearDict[year]
   flist = maxbtag_files if private else maxbtag_4b_files
   mpoint = [f for f in flist if str(mx) in f and str(my) in f][0] # find filename with both mx and my in it
   if len(mpoint) < 0: raise FileNotFoundError(f'File not found: {mpoint}')
   if suffix != '': suffix = '_' + suffix
   mpoint = f"NMSSM/{mpoint}{suffix}" if private else f"Official_NMSSM/{mpoint}"
   return f"{base}/{year}/{selection}/{mpoint}/ntuple.root"

def get_NMSSM_list(run='Summer2018UL', jets='maxbtag_4b', private=False):
   location = f"{base}/{run}/{jets}/"
   if not private: location += "Official_"
   location += "NMSSM"
   flist = os.listdir(location)
   flist = [f"{location}/{f}/ntuple.root" for f in flist if 'NMSSM' in f]
   return flist

def get_nocuts(mx=700, my=400, run='Summer2018UL'):
   return f"{base}/{run}/cutflow_studies/nocuts/NMSSM/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}/ntuple.root"

def get_trigger(mx=700, my=400, run='Summer2018UL'):
   return f"{base}/{run}/cutflow_studies/trigger/NMSSM/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}/ntuple.root"

def get_presel(mx=700, my=400, run='Summer2018UL'):
   return f"{base}/{run}/cutflow_studies/presel/NMSSM/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}/ntuple.root"

def get_data(selection='Summer2018UL', jets='maxbtag_4b'):
   return f"{base}/{selection}/{jets}/{data_path[run]}"

def get_qcd_enriched(selection, run='Summer2018UL'):
   QCD_bEn_Ht_100to200   = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT100to200_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_200to300   = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT200to300_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_300to500   = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT300to500_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_500to700   = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT500to700_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_700to1000  = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT700to1000_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_1000to1500 = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_1500to2000 = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_2000toInf  = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"

   QCD_bEn_List = [QCD_bEn_Ht_100to200,QCD_bEn_Ht_200to300,QCD_bEn_Ht_300to500, QCD_bEn_Ht_500to700, QCD_bEn_Ht_700to1000, QCD_bEn_Ht_1000to1500,QCD_bEn_Ht_1500to2000,QCD_bEn_Ht_2000toInf]

   return QCD_bEn_List

def get_qcd_bgen(selection, run='Summer2018UL'):
   QCD_bGf_Ht_100to200   = f"{base}/{run}/{selection}/QCD/QCD_HT100to200_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_200to300   = f"{base}/{run}/{selection}/QCD/QCD_HT200to300_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_300to500   = f"{base}/{run}/{selection}/QCD/QCD_HT300to500_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_500to700   = f"{base}/{run}/{selection}/QCD/QCD_HT500to700_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_700to1000  = f"{base}/{run}/{selection}/QCD/QCD_HT700to1000_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_1000to1500 = f"{base}/{run}/{selection}/QCD/QCD_HT1000to1500_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_1500to2000 = f"{base}/{run}/{selection}/QCD/QCD_HT1500to2000_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_2000toInf  = f"{base}/{run}/{selection}/QCD/QCD_HT2000toInf_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"

   QCD_bGf_List = [QCD_bGf_Ht_100to200,QCD_bGf_Ht_200to300, QCD_bGf_Ht_300to500, QCD_bGf_Ht_500to700, QCD_bGf_Ht_700to1000, QCD_bGf_Ht_1000to1500,QCD_bGf_Ht_1500to2000,QCD_bGf_Ht_2000toInf]

   return QCD_bGf_List

def get_qcd_pt(selection, run='Summer2018UL'):
   QCD_HT1000to1500 = f"{base}/{run}/{selection}/QCD/QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/ntuple.root"
   QCD_HT100to200   = f"{base}/{run}/{selection}/QCD/QCD_HT100to200_TuneCP5_PSWeights_13TeV-madgraph-pythia8/ntuple.root"
   QCD_HT1500to2000 = f"{base}/{run}/{selection}/QCD/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/ntuple.root"
   QCD_HT2000toInf  = f"{base}/{run}/{selection}/QCD/QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8/ntuple.root"
   QCD_HT200to300   = f"{base}/{run}/{selection}/QCD/QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8/ntuple.root"
   QCD_HT300to500   = f"{base}/{run}/{selection}/QCD/QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/ntuple.root"
   QCD_HT500to700   = f"{base}/{run}/{selection}/QCD/QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8/ntuple.root"
   QCD_HT50to100    = f"{base}/{run}/{selection}/QCD/QCD_HT50to100_TuneCP5_PSWeights_13TeV-madgraph-pythia8/ntuple.root"
   QCD_HT700to1000  = f"{base}/{run}/{selection}/QCD/QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/ntuple.root"

   QCD_List = [QCD_HT1000to1500, QCD_HT100to200, QCD_HT1500to2000, QCD_HT2000toInf, QCD_HT200to300, QCD_HT300to500, QCD_HT500to700, QCD_HT50to100, QCD_HT700to1000]

   return QCD_List

def get_qcd_list(selection, run='Summer2018UL'):
   QCD_bEn_Ht_100to200   = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT100to200_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_200to300   = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT200to300_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_300to500   = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT300to500_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_500to700   = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT500to700_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_700to1000  = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT700to1000_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_1000to1500 = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_1500to2000 = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bEn_Ht_2000toInf  = f"{base}/{run}/{selection}/QCD/QCD_bEnriched_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"

   QCD_bEn_List = [QCD_bEn_Ht_100to200,QCD_bEn_Ht_200to300,QCD_bEn_Ht_300to500, QCD_bEn_Ht_500to700, QCD_bEn_Ht_700to1000, QCD_bEn_Ht_1000to1500,QCD_bEn_Ht_1500to2000,QCD_bEn_Ht_2000toInf]

   QCD_bGf_Ht_100to200   = f"{base}/{run}/{selection}/QCD/QCD_HT100to200_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_200to300   = f"{base}/{run}/{selection}/QCD/QCD_HT200to300_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_300to500   = f"{base}/{run}/{selection}/QCD/QCD_HT300to500_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_500to700   = f"{base}/{run}/{selection}/QCD/QCD_HT500to700_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_700to1000  = f"{base}/{run}/{selection}/QCD/QCD_HT700to1000_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_1000to1500 = f"{base}/{run}/{selection}/QCD/QCD_HT1000to1500_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_1500to2000 = f"{base}/{run}/{selection}/QCD/QCD_HT1500to2000_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
   QCD_bGf_Ht_2000toInf  = f"{base}/{run}/{selection}/QCD/QCD_HT2000toInf_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"

   QCD_bGf_List = [QCD_bGf_Ht_100to200,QCD_bGf_Ht_200to300, QCD_bGf_Ht_300to500, QCD_bGf_Ht_500to700, QCD_bGf_Ht_700to1000, QCD_bGf_Ht_1000to1500,QCD_bGf_Ht_1500to2000,QCD_bGf_Ht_2000toInf]

   QCD_B_List = QCD_bEn_List + QCD_bGf_List
   return QCD_B_List

def get_ttbar(selection, run='Summer2018UL', gnn=False):
   TTJets = f"{base}/{run}/{selection}/TTJets/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/ntuple.root"
   if gnn: TTJets = f"{base}/{run}/{selection}/TTJets/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/ntuple.root"
   return TTJets

def get_qcd_ttbar(selection, run='Summer2018UL', enriched=False, bgen=False, gnn=False):
   if enriched: return get_qcd_enriched(selection, run) + [get_ttbar(selection, run, gnn=gnn)]
   if bgen: return get_qcd_bgen(selection, run) + [get_ttbar(selection, run, gnn=gnn)]
   return get_qcd_list(selection, run) + [get_ttbar(selection, run, gnn=gnn)]
