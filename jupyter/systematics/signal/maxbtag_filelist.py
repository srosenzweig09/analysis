"""
Writes txt files containing the root file locations for the nominal and systematic central signal samples, as well as the privately-produced samples used to calculate the correction ratio for the b tag sf.
"""

import subprocess, sys

def get_mx(out):
    return int(out.split('/')[-2].split('-')[1].split('_')[0])

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

# base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM'
# cmd = f"ls {base}/*/ntuple.root"
# output = subprocess.check_output(cmd, shell=True).decode('utf-8').split('\n')
# output = [f"{out}\n" for out in output if 'NMSSM' in out]
# # print(output)
# ind = [i for i,out in enumerate(output) if 'MX_700_MY_400' in out and '_2M' not in out][0]
# output.pop(ind)
# output[-1] = output[-1].replace('\n', '')

# with open("sf_files.txt", "w") as f:
#     f.writelines(output)



base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM'
cmd = f"ls {base}/*/ntuple.root"
output = subprocess.check_output(cmd, shell=True).decode('utf-8').split('\n')
output = [f"{out}\n" for out in output if 'NMSSM' in out]
output = [out for out in output if get_mx(out) < 1300]
# output = [out.replace('/eos/uscms', '') for out in output]
# output = [out.replace('/ntuple.root', '') for out in output]
output[-1] = output[-1].replace('\n', '')

with open("sig_files.txt", "w") as f:
    f.writelines(output)




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

# with open("jec_files.txt", "w") as f:
#     # f.write(samples)
#     f.writelines(files)
