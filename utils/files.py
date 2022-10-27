import os

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
    [1200,300], [1200,400], [1200,500], [1200,600], [1200,700], [1200,800], [1200,900],
    ]

fnal_root = "root://cmseos.fnal.gov/"
base = '/eos/uscms/store/user/srosenzw/sixb/ntuples'
current_base = f"{base}/Summer2018UL/bias/"
data_path = 'JetHT_Data_UL/JetHT_Run2018_full/ntuple.root'
config = 'config/sphereConfig.cfg'

def get_NMSSM(mx, my, run='Summer2018UL', jets='bias'):
   return f"{base}/{run}/{jets}/NMSSM/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}/ntuple.root"

def get_presel(mx=700, my=400, run='Summer2018UL'):
   return f"{base}/{run}/NMSSM_cutflow_studies/presel/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}/ntuple.root"

def get_trigger(mx=700, my=400, run='Summer2018UL'):
   return f"{base}/{run}/NMSSM_trigger/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}/ntuple.root"

def get_nocuts(mx=700, my=400, run='Summer2018UL', study='nocuts'):
   return f"{base}/{run}/NMSSM_nocuts/NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}/ntuple.root"

def get_data(run='Summer2018UL', jets='bias', btag=False):
   if btag: jets = jets + '_maxbtag'
   return f"{base}/{run}/{jets}/{data_path}"

def get_signal_list(run='Summer2018UL', jets='bias'):
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

# class FileLocations:

#    def __init__(self, run='Summer2018UL', pair='dHHH_pairs'):

#       self.run = run
#       self.pair = pair

#       self.base = f"/eos/uscms/store/user/srosenzw/sixb/sixb_ntuples/{self.run}/{self.pair}"

#       # self.no_cuts = "/eos/uscms/store/user/srosenzw/sixb/studies/NMSSM_nocuts/NMSSM_XYH_YToHH_6b_MX_700_MY_400/ntuple.root"

#       self.data = f"{self.base}/JetHT_Data_UL/JetHT_Run2018_full/ntuple.root"
#       if not os.path.isfile(self.data):
#          raise FileNotFoundError(f'File not found: {self.data}')

#       self.NMSSM = []
#       for mx,my in mx_my_masses:
#          mxmy_name = f'NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}'
#          mxmy_loc  = f'{self.base}/NMSSM/{mxmy_name}/ntuple.root'
#          if not os.path.isfile(mxmy_loc):
#                # raise FileNotFoundError(f'File not found: {mxmy_loc}')
#                print(f"Skipping missing file: {mxmy_loc}")
#                continue
#          setattr(self, mxmy_name, mxmy_loc)
#          self.NMSSM.append(mxmy_loc)

#    def get_NMSSM(self, mx, my):
#       mxmy_name = f'NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}'
#       mxmy_loc  = f'{self.base}/NMSSM/{mxmy_name}/ntuple.root'
#       return getattr(self, mxmy_name)

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

def get_ttbar(selection, run='Summer2018UL'):
   TTJets = f"{base}/{run}/{selection}/TTJets/TTJets/ntuple.root"
   return TTJets

def get_qcd_ttbar(selection, run='Summer2018UL'):
   return get_qcd_list(selection, run) + [get_ttbar(selection, run)]


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