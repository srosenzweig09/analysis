from utils.analysis.datacard import DataCard
import os
from utils.analysis.feyn import model_name

def get_mx_my(mass):
    mx = int(mass.split('_')[1])
    my = int(mass.split('_')[3].split('.')[0])
    return [mx,my]

dirname = f'/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/combine/feynnet/{model_name}'

masses = [d for d in os.listdir(dirname) if 'MX_' in d]
masses = [get_mx_my(mass) for mass in masses]

for mx,my in masses:
    dc = DataCard(f'MX_{mx}_MY_{my}')
    dc.write_no_systematics()

# dc = DataCard('MX_500_MY_300')
# dc.write()
# dc = DataCard('MX_1000_MY_800')
# dc.write()
# dc = DataCard('MX_850_MY_350')
# dc.write()
