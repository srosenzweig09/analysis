import subprocess

base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM'

cmd = f"ls {base}/*/ntuple.root"

output = subprocess.check_output(cmd, shell=True).decode('utf-8').split('\n')
output = [f"{out}\n" for out in output if 'NMSSM' in out]
print(output)

with open("sf_files.txt", "w") as f:
    f.writelines(output)
