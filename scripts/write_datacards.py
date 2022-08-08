from utils.analysis.data import DataCard
import shlex
import subprocess

pairs = 'dHHH'
jets = 'bias'
scheme = 'sphere'

cmd = f'ls combine/{pairs}/{jets}/{scheme}/root/'
output = subprocess.check_output(shlex.split(cmd))
sub_dirs = output.decode("utf-8").split('\n')[:-2]

for file in sub_dirs:
   filename = file.split('.')[0]
   print(f"[INFO] Processing {filename}")
   datacard = DataCard(filename)
   print()

   # cmd = f'cat combine/{pairs}/{jets}/{scheme}/datacards/{filename}.txt'
   # subprocess.call(shlex.split(cmd))