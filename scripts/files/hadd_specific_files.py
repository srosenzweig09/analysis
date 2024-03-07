
import subprocess, shlex

filepath = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM/NMSSM_XYH_YToHH_6b_'

filelist = [
    'MX_1000_MY_250_2M',
    'MX_1000_MY_800_2M',
    'MX_1100_MY_250_2M',
    'MX_1100_MY_900_2M',
    'MX_1200_MY_250_2M',
    'MX_1200_MY_1000_2M',
    'MX_900_MY_250_2M',
    'MX_900_MY_700_2M',
    'MX_800_MY_250_2M',
    'MX_800_MY_600_2M',
    'MX_700_MY_250_2M',
    'MX_600_MY_250_4M',
    'MX_500_MY_250_10M',
    'MX_400_MY_250_10M',
    ]

for file in filelist:
    print(f"-------------------- HADDING {file} --------------------")
    cmd = f"ls {filepath}{file}/output"
    output = subprocess.check_output(shlex.split(cmd))
    output = output.decode("utf-8").split('\n')
    infiles = [f"{filepath}{file}/output/{x}" for x in output if 'root' in x]
    infiles = ' '.join(infiles)

    cmd = f"hadd -f {filepath}{file}/ntuple.root {infiles}"
    subprocess.run(shlex.split(cmd))