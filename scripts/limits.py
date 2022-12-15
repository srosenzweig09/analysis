import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import shlex, subprocess
import sys
 
def call(cmd, **kwargs):
  return subprocess.call(shlex.split(cmd), **kwargs)
 
def get(cmd, **kwargs):
   output = subprocess.check_output(shlex.split(cmd), **kwargs)
   return output.decode("UTF-8")

# jets = 'btag'
# path = f'/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/limits/{jets}_sphere.txt'

path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/limits/btag_pt_limits.txt'

with open(path) as f:
   lines = f.readlines()

lines[1:] = [line[:-2] for line in lines[1:]]

cols = lines[0][:-1].split(' ')
data = lines[1:]
data = [line.split(' ') for line in data]

for i,line in enumerate(data):
   if '' in line: continue
   if line == '\n': continue
   numbers = [float(num) for num in line]
   data[i] = numbers

# data = [line.split[' '] for line in data]
# print(data)

df = DataFrame(data, columns=cols)

data = np.hstack([line for line in data]).reshape(len(data),len(data[0]))
my = data[:,1]

df = df.astype({key:int for key in cols[:2]})
df = df.set_index(cols[:2])
df = df.sort_index()

print(df)

fig, ax = plt.subplots()
for MY in np.unique(my):
   limits = df.xs(MY, level='my')['mean']
   limits.plot(ax=ax, xlabel=r'M$_\mathrm{X}$ [GeV]', ylabel=r'$\sigma(\mathrm{X}\rightarrow\mathrm{Y(HH)H}\rightarrow\mathrm{6b})$ [fb]')


labels = [r"M$_\mathrm{Y}$ = " + f"{int(mass)} GeV" for mass in np.unique(my)]
ax.legend(labels=labels)
ax.set_title(rf"59.7 fb$^{{-1}}$ (13 TeV, 2018)")
# ax.legend()
# ax.set_ylabel(r"$\sigma(X\rightarrow Y(HH)H \rightarrow 6b$ [fb]")
# ax.set_xlabel(r"M$_\text{X}$ [GeV]")


fig.savefig(f"limits/btag_pt/limits_btag_pt.pdf", bbox_inches='tight')
# plt.show()