"""
Writes txt files containing the root file locations for the nominal and systematic central signal samples, as well as the privately-produced samples used to calculate the correction ratio for the b tag sf.
"""
import re, os
# import sys

def get_central_mx(out):
    return int(out.split('-')[1].split('_')[0])

def get_private_mx(out):
    # print(out)
    return int(re.search('MX_(.*)_MY_', out).group(1))
    # return int(out.split('/')[-2].split('_')[1])

def fname(out):
    return out.split('/')[14]

def syst(out):
    return out.split('/')[12]

def var(out):
    return out.split('/')[13]

def strip_ends(out):
    tmp = out.split('/eos/uscms')[1]
    # tmp = tmp.split('\n')[0]
    return tmp

# Central Samples ######################################
base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM'
# cmd = f"ls {base}/*/ntuple.root"
output = os.listdir(base)
output = [out for out in output if 'NMSSM' in out]
masses = [f"{base}/{out}/ntuple.root\n" for out in output if get_central_mx(out) < 1300]
with open("filelists/central.txt", "w") as f:
    f.writelines(masses)

masses = [f"root://cmseos.fnal.gov//store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/{out}/ntuple.root\n" for out in output if get_central_mx(out) < 1300] # with spaces for FeynNet prediction
with open("filelists/central_feynman.txt", "w") as f:
    f.writelines(masses)

# Private Samples ######################################
base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM'
output = os.listdir(base)
output = [out for out in output if 'NMSSM' in out]
output = [f"{base}/{out}/ntuple.root\n" for out in output if get_private_mx(out) < 1300]
with open("filelists/private.txt", "w") as f:
    f.writelines(output)

# Systematic Samples ######################################
base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/syst'
systematics = os.listdir(base)
systematics = [f"{base}/{syst}/" for syst in systematics]
for syst in systematics:
    masses = os.listdir(f"{syst}/up")
    masses = [out for out in masses if 'NMSSM' in out]
    masses = [f"{base}/{out}/ntuple.root\n" for out in masses if get_central_mx(out) < 1300]
    break
# output = [out for out in output if 'NMSSM' in out]
# output = [f"{base}/{out}/ntuple.root\n" for out in output if get_private_mx(out) < 1300]



# cmd = f"ls {base}/syst/*/*/*/ntuple.root"
# output = subprocess.check_output(cmd, shell=True).decode('utf-8').split('\n')
# # output = [f"{out}" for out in output if 'NMSSM' in out and 'jer_pt' in out.lower()]
# # output[-1] = output[-1].replace('\n', '')

# samples = []
# files = [""]
# for out in output:
#     try: files.append(out.split('/eos/uscms')[1].split('/ntuple.root')[0]+'\n')
#     except IndexError: print(out)

#     # fn = f"syst/{syst(out)}/{var(out)}/{fname(out)}.root"
#     # floc = strip_ends(out)
#     # samples.append(f"{fn}:{floc}")
#     # files.append(f"{floc.removesuffix('/ntuple.root')}\n")
#     # files.append(strip_ends(out))
# # samples = ' '.join(samples)

# # print(files)

# with open("filelists/jec_files.txt", "w") as f:
#     # f.write(samples)
#     f.writelines(files)
