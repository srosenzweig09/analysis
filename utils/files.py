import os
import subprocess, shlex

run_conditions  = ['Summer2018UL']
pairing_schemes = ['bias', 'btag']
studies         = ['nocuts', 'presel']

mx_my_masses = [
    [ 450,300],
    [ 500,300],
    [ 600,300], [ 600,400],
    [ 700,300], [ 700,400], [ 700,500],
    [ 800,300], [ 800,400], [ 800,500], [ 800,600],
    [ 900,300], [ 900,400], [ 900,500], [ 900,600], [ 900,700],
    [1000,300], [1000,400], [1000,500], [1000,600], [1000,700], [1000,800],
    [1100,300], [1100,400], [1100,500], [1100,600], [1100,700], [1100,800], [1100,900],
    [1200,300], [1200,400], [1200,500], [1200,600], [1200,700], [1200,800], [1200,900], [1200,1000]
    ]

fnal_root = "root://cmseos.fnal.gov/"
base = '/eos/uscms/store/user/srosenzw/sixb/ntuples'
# data_path = 'JetHT_Data_UL/JetHT_Run2018_full/ntuple.root'
data_path = {
   'Summer2018UL' : 'JetHT_Data_UL/ntuple.root',
   'Summer2017UL' : 'BTagCSV/ntuple.root',
}
config = 'config/sphereConfig.cfg'

def get_data(run='Summer2018UL', jets='btag_pt'):
   return f"{base}/{run}/{jets}/{data_path}"

def get_NMSSM(mx=700, my=400, run='Summer2018UL', jets='maxbtag_4b', cut=None, append='', private=False):
   if cut is not None: jets = ''
   if append != '': append = '_' + append
   file_dict = {
      'presel' : 'cutflow_studies/presel',
      'trigger' : 'cutflow_studies/trigger',
      'presel' : 'cutflow_studies/presel',
   }
   if private: return f"{base}/{run}/{jets}{file_dict.get(cut, '')}/NMSSM/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}{append}/ntuple.root"
   return f"{base}/{run}/{jets}{file_dict.get(cut, '')}/Official_NMSSM/NMSSM_XToYHTo6B_MX-{mx}_MY-{my}{append}_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"

def get_NMSSM_list(run='Summer2018UL', jets='maxbtag_4b', cut=None, append='', private=False):
   location = f"{base}/{run}/{jets}/"
   if not private: location += "Official_"
   location += "NMSSM"
   print(location)
   output = subprocess.check_output(shlex.split(f"ls {location}")).decode("UTF-8")
   output  = [f"{location}/{out}/ntuple.root" for out in output.split('\n') if 'NMSSM' in out]
   return output


def get_nocuts(mx=700, my=400, run='Summer2018UL', study='nocuts'):
   return f"{base}/{run}/cutflow_studies/nocuts/NMSSM/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}/ntuple.root"

def get_trigger(mx=700, my=400, run='Summer2018UL'):
   return f"{base}/{run}/cutflow_studies/trigger/NMSSM/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}/ntuple.root"

def get_presel(mx=700, my=400, run='Summer2018UL'):
   return f"{base}/{run}/cutflow_studies/presel/NMSSM/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}/ntuple.root"

def get_data(run='Summer2018UL', jets='maxbtag_4b'):
   return f"{base}/{run}/{jets}/{data_path[run]}"

def get_signal_list(run='Summer2018UL', jets='maxbtag_4b'):
   NMSSM = []
   base = f"{base}/{run}/{jets}"
   for mx,my in mx_my_masses:
      mxmy_name = f'NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}'
      mxmy_loc  = f'{base}/NMSSM/{mxmy_name}/ntuple.root'
      if not os.path.isfile(mxmy_loc):
            # raise FileNotFoundError(f'File not found: {mxmy_loc}')
            print(f"Skipping missing file: {mxmy_loc}")
            continue
      NMSSM.append(mxmy_loc)
   return NMSSM

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


# studies = "/eos/uscms/store/user/srosenzw/sixb/studies"
# tag = "_nocuts"
# NMSSM_MX_450_MY_300_studies = f"{studies}/NMSSM{tag}/NMSSM_XYH_YToHH_6b_MX_450_MY_300/ntuple.root"
# NMSSM_MX_500_MY_300_studies = f"{studies}/NMSSM{tag}/NMSSM_XYH_YToHH_6b_MX_500_MY_300/ntuple.root"
# NMSSM_MX_600_MY_300_studies = f"{studies}/NMSSM{tag}/NMSSM_XYH_YToHH_6b_MX_600_MY_300/ntuple.root"
# NMSSM_MX_600_MY_400_studies = f"{studies}/NMSSM{tag}/NMSSM_XYH_YToHH_6b_MX_600_MY_400/ntuple.root"
# NMSSM_MX_700_MY_300_studies = f"{studies}/NMSSM{tag}/NMSSM_XYH_YToHH_6b_MX_700_MY_300/ntuple.root"
# NMSSM_MX_700_MY_400_studies = f"{studies}/NMSSM{tag}/NMSSM_XYH_YToHH_6b_MX_700_MY_400/ntuple.root"
# # NMSSM_MX_700_MY_400_studies = f"{studies}/NMSSM{tag}/NMSSM_XYH_YToHH_6b_MX_700_MY_400_10M/ntuple.root"
# NMSSM_MX_700_MY_500_studies = f"{studies}/NMSSM{tag}/NMSSM_XYH_YToHH_6b_MX_700_MY_500/ntuple.root"
# NMSSM_List_studies = [NMSSM_MX_450_MY_300_studies,NMSSM_MX_500_MY_300_studies,NMSSM_MX_600_MY_300_studies,NMSSM_MX_600_MY_400_studies,NMSSM_MX_700_MY_300_studies,NMSSM_MX_700_MY_400_studies,NMSSM_MX_700_MY_500_studies]