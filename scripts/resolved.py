from utils.analysis.signal import SixB
from utils.bashUtils import check_output

base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM'
cmd = f'ls {base}'
list_of_dirs = check_output(cmd)

for cdir in list_of_dirs:
    