import numpy as np
import re, subprocess, shlex
from pandas import DataFrame

cmd = 'ls data/btag'
output = subprocess.check_output(shlex.split(cmd)).decode('UTF-8').split('\n')[:-1]
output = [out for out in output if out.endswith('.root')]

sf_masses = [[int(out[3:out.find('_MY_')]), int(out[out.find('_MY_')+4:out.find('.root')])] for out in output]

sf_dict = {out[0]:[] for out in sf_masses}
for out in sf_masses:
    sf_dict[out[0]].append(out[1])
# print(sf_dict)

# sf_df = DataFrame.from_dict(sf_dict, orient='index')
# print(sf_df)

# print(sf_masses)

cmd = 'ls /eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM/'
output = subprocess.check_output(shlex.split(cmd)).decode('UTF-8').split('\n')[:-1]
output = [out for out in output if 'NMSSM' in out]

eos_masses = []
for out in output:
    eos_masses.append([int(out.split('_')[5]), int(out.split('_')[7])])
# print(eos_masses)

eos_dict = {out[0]:[] for out in eos_masses}
for out in eos_masses:
    # print(out[0], out[1])
    if out[1] not in eos_dict[out[0]]: eos_dict[out[0]].append(out[1])
# print(eos_dict)
# eos_df = DataFrame.from_dict(eos_dict, orient='index')
# print(eos_df)

for mx in eos_dict.keys():
    my = np.array(eos_dict[mx])
    check = sf_dict[mx]
    mask = np.isin(my, check)
    print(f"{mx} : {my[~mask]}")