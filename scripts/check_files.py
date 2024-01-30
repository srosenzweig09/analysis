# parallel -j 10 "python scripts/check_files.py {}" ::: $(cat sig_files.txt) --eta
"""
"""

from argparse import ArgumentParser
from colorama import Fore, Style
from utils.analysis.signal import SixB

parser = ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

systematics = ['JERpt', 'bJER','Absolute_2018', 'Absolute', 'BBEC1', 'BBEC1_2018', 'EC2', 'EC2_2018', 'FlavorQCD', 'HF', 'HF_2018', 'RelativeBal', 'RelativeSample_2018']

def raiseError(fname):
    print(f"{Fore.RED}[FAILED]{Style.RESET_ALL} Cannot open file:\n{fname}")

fname = args.filename

try: tree = SixB(fname)
except: raiseError(fname)

for systematic in systematics:
    tmp = fname.split('/')
    tmp.insert(11, "syst")
    tmp.insert(12, systematic)
    tmp.insert(13, "up")
    up = '/'.join(tmp)
    try: up = SixB(up)
    except: raiseError(up)

    tmp[13] = "down"
    down = '/'.join(tmp)
    try: down = SixB(down)
    except: raiseError(down)

