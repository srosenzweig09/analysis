import os

run_conditions  = ['Summer2018UL']
pairing_schemes = ['dHHH_pairs','mH_pairs']

default_run = run_conditions[0]
default_pair = pairing_schemes[0]

mx_my_masses = [
    [ 450,300],
    [ 500,300],
    [ 600,300],
    [ 600,400],
    [ 700,300],
    [ 700,400],
    [ 700,500],
    [1000,300],
    [1000,700],
    ]

class FileLocations:
    # mx_my_masses = [
    #     [ 450,300],
    #     [ 500,300],
    #     [ 600,300],
    #     [ 600,400],
    #     [ 700,300],
    #     [ 700,400],
    #     [ 700,500],
    #     [1000,300],
    #     [1000,700],
    #     ]

    mx_my_masses = mx_my_masses

    run_conditions  = run_conditions
    pairing_schemes = pairing_schemes

    def __init__(self, run='default', pair='default', warn=True):

        self.warn = warn
        self.check_initializations(run, pair)
        
        self.base = f"/eos/uscms/store/user/srosenzw/sixb/sixb_ntuples/{self.run}/{self.pair}"

        self.data = f"{self.base}/JetHT_Data_UL/JetHT_Run2018_full/ntuple.root"
        if not os.path.isfile(self.data):
            raise FileNotFoundError(f'File not found: {self.data}')

        self.NMSSM = []
        for mx,my in self.mx_my_masses:
            mxmy_name = f'NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}'
            mxmy_loc  = f'{self.base}/NMSSM/{mxmy_name}/ntuple.root'
            if not os.path.isfile(mxmy_loc) and warn:
                # raise FileNotFoundError(f'File not found: {mxmy_loc}')
                print(f"Skipping missing file: {mxmy_loc}")
                continue
            setattr(self, mxmy_name, mxmy_loc)
            self.NMSSM.append(mxmy_loc)

    def get_NMSSM(self, mx, my):
        mxmy_name = f'NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}'
        mxmy_loc  = f'{self.base}/NMSSM/{mxmy_name}/ntuple.root'
        return getattr(self, mxmy_name)

    def check_initializations(self, run, pair):
        if run == 'default' and self.warn:
            print(f"Using default run: {run}")
        if pair == 'default' and self.warn:
            print(f"Using default pair: {pair}")
        if run != 'default' and run not in self.run_conditions:
            raise KeyError(f'Run "{run}" is not a valid  option. Please use one of the following keys:\n{run_conditions.keys()}')
        if pair != 'default' and pair not in self.pairing_schemes:
            raise KeyError(f'Run "{pair}" is not a valid  option. Please use one of the following keys:\n{pairing_schemes.keys()}')
        
        self.run = default_run
        self.pair = default_pair


# NMSSM_MX_450_MY_300 = f"{base}/NMSSM/NMSSM_XYH_YToHH_6b_MX_450_MY_300/ntuple.root"
# NMSSM_MX_500_MY_300 = f"{base}/NMSSM/NMSSM_XYH_YToHH_6b_MX_500_MY_300/ntuple.root"
# NMSSM_MX_600_MY_300 = f"{base}/NMSSM/NMSSM_XYH_YToHH_6b_MX_600_MY_300/ntuple.root"
# NMSSM_MX_600_MY_400 = f"{base}/NMSSM/NMSSM_XYH_YToHH_6b_MX_600_MY_400/ntuple.root"
# NMSSM_MX_700_MY_300 = f"{base}/NMSSM/NMSSM_XYH_YToHH_6b_MX_700_MY_300/ntuple.root"
# NMSSM_MX_700_MY_400 = f"{base}/NMSSM/NMSSM_XYH_YToHH_6b_MX_700_MY_400/ntuple.root"
# # NMSSM_MX_700_MY_400 = f"{base}/NMSSM/NMSSM_XYH_YToHH_6b_MX_700_MY_400_10M/ntuple.root"
# NMSSM_MX_700_MY_500 = f"{base}/NMSSM/NMSSM_XYH_YToHH_6b_MX_700_MY_500/ntuple.root"
# NMSSM_MX_1000_MY_300 = f"{base}/NMSSM/NMSSM_XYH_YToHH_6b_MX_1000_MY_300/ntuple.root"
# NMSSM_MX_1000_MY_700 = f"{base}/NMSSM/NMSSM_XYH_YToHH_6b_MX_1000_MY_700/ntuple.root"

# NMSSM_List = [NMSSM_MX_450_MY_300,NMSSM_MX_500_MY_300,NMSSM_MX_600_MY_300,NMSSM_MX_600_MY_400,NMSSM_MX_700_MY_300,NMSSM_MX_700_MY_400,NMSSM_MX_700_MY_500,NMSSM_MX_1000_MY_300,NMSSM_MX_1000_MY_700]

# JetHT_Data_UL = f"{base}/JetHT_Data_UL/JetHT_Run2018_full/ntuple.root"


# QCD_bEn_Ht_100to200   = f"{base}/QCD/QCD_bEnriched_HT100to200_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bEn_Ht_200to300   = f"{base}/QCD/QCD_bEnriched_HT200to300_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bEn_Ht_300to500   = f"{base}/QCD/QCD_bEnriched_HT300to500_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bEn_Ht_500to700   = f"{base}/QCD/QCD_bEnriched_HT500to700_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bEn_Ht_700to1000  = f"{base}/QCD/QCD_bEnriched_HT700to1000_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bEn_Ht_1000to1500 = f"{base}/QCD/QCD_bEnriched_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bEn_Ht_1500to2000 = f"{base}/QCD/QCD_bEnriched_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bEn_Ht_2000toInf  = f"{base}/QCD/QCD_bEnriched_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"

# QCD_bEn_List = [QCD_bEn_Ht_100to200,QCD_bEn_Ht_200to300,QCD_bEn_Ht_300to500, QCD_bEn_Ht_500to700, QCD_bEn_Ht_700to1000, QCD_bEn_Ht_1000to1500,QCD_bEn_Ht_1500to2000,QCD_bEn_Ht_2000toInf]

# QCD_bGf_Ht_100to200   = f"{base}/QCD/QCD_HT100to200_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bGf_Ht_200to300   = f"{base}/QCD/QCD_HT200to300_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bGf_Ht_300to500   = f"{base}/QCD/QCD_HT300to500_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bGf_Ht_500to700   = f"{base}/QCD/QCD_HT500to700_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bGf_Ht_700to1000  = f"{base}/QCD/QCD_HT700to1000_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bGf_Ht_1000to1500 = f"{base}/QCD/QCD_HT1000to1500_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bGf_Ht_1500to2000 = f"{base}/QCD/QCD_HT1500to2000_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
# QCD_bGf_Ht_2000toInf  = f"{base}/QCD/QCD_HT2000toInf_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"

# QCD_bGf_List = [QCD_bGf_Ht_100to200,QCD_bGf_Ht_200to300, QCD_bGf_Ht_300to500, QCD_bGf_Ht_500to700, QCD_bGf_Ht_700to1000, QCD_bGf_Ht_1000to1500,QCD_bGf_Ht_1500to2000,QCD_bGf_Ht_2000toInf]

# QCD_B_List = QCD_bEn_List + QCD_bGf_List

# TTJets = f"{base}/TTJets/TTJets/ntuple.root"


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