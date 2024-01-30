# parallel -j 4 "python scripts/file_handling/hadd_private.py {}" ::: $(cat sig_files.txt) --eta

from argparse import ArgumentParser
import subprocess
from colorama import Fore, Style

parser = ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

filename = args.filename
base = filename.split('ntuple.root')[0] 
base = base.split('eos/uscms')[1]
signal = base.split('/')[-2]

# print(f"hadd -f {signal}.root {base}output/*")

print("[INFO] Processing signal: {}{}{}".format(Fore.MAGENTA, signal, Style.RESET_ALL))
cmd = f"hadd -f {signal}.root `xrdfs root://cmseos.fnal.gov ls -u {base}output | grep '\.root'`"
print(".. hadding files into local dir")
subprocess.call(cmd, shell=True)
cmd = f"xrdcp -f {signal}.root root://cmseos.fnal.gov/{base}ntuple.root"
print(".. copying file to eos")
subprocess.call(cmd, shell=True)
cmd = f"rm {signal}.root"
print(".. removing local file")
subprocess.call(cmd, shell=True)