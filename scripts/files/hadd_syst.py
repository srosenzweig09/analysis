# parallel -j 15 "python scripts/file_handling/hadd_syst.py {}" ::: $(cat jec_files.txt) --eta

from argparse import ArgumentParser
import subprocess

parser = ArgumentParser()
parser.add_argument("dirname")
args = parser.parse_args()

dirname = args.dirname
fname = dirname.split('/')[-1]

cmd = f"hadd {fname}.root `xrdfs root://cmseos.fnal.gov ls -u {dirname}/output | grep '\.root'`"
print(".. hadding files into local dir")
subprocess.call(cmd, shell=True)
cmd = f"xrdcp -f {fname}.root root://cmseos.fnal.gov/{dirname}/ntuple.root"
print(".. copying file to eos")
subprocess.call(cmd, shell=True)
subprocess.call(f"rm {fname}.root", shell=True)

