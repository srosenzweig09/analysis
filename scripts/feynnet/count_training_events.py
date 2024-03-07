import uproot as up
import subprocess, shlex
from tqdm import tqdm

fileloc = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM'
cmd = f"ls {fileloc}"
output = subprocess.check_output(shlex.split(cmd))
output = output.decode('utf-8')
output = output.split('\n')
output = [f"{fileloc}/{out}/fully_res_ntuple.root" for out in output if 'NMSSM' in out]

n = 0
for out in tqdm(output):
    tree = up.open(f"{out}:sixBtree")
    n += len(tree['X_m'].array())

tree = up.open("/eos/uscms/store/user/srosenzw/sixb/ntuples/2018_gnn_training/feynnet_training/background.root:sixBtree")
n += len(tree['X_m'].array())

print(n)