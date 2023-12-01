from matplotlib.backends.backend_pdf import PdfPages
from utils.analysis import sixb_from_gnn
import matplotlib.pyplot as plt
from rich import print as rprint
from rich.console import Console
console = Console()
from tqdm import tqdm
import subprocess, shlex
from utils.plotter import Hist
import numpy as np

from utils.analysis.gnn import model_path
model_dir = model_path.split('/')[-2]
model_savein = f'plots/feynnet/{model_dir}'
model_dir

info_dict = {}

# load the signal
base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM'
cmd = f"ls {base}"
output = subprocess.check_output(shlex.split(cmd))
output = output.decode('UTF-8')
output = output.split('\n')
output = [f"{base}/{out}/ntuple.root" for out in output if out.startswith('NMSSM')]

signal = []
with console.status("[bold][green]Loading signal...") as status:
    # with suppress_stdout():
    signal = [sixb_from_gnn(out) for out in output]
    # for out in output:
        # try: signal.append(SixB(out))
        # except: print(f"[red]Failed to load: {out}")

print("..generating plots of top score difference")
with PdfPages(f"/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/plots/feynnet/{model_dir}/score_diff_max_nmax.pdf") as pdf:
    for sig in tqdm(signal):
        rprint(f"Processing: mx = {sig.mx}, my = {sig.my}")
        # sig.initialize_gen()

        correct_mask = sig.H1_correct & sig.H2_correct

        fig, ax = plt.subplots()
        n = Hist(sig.max_diff[correct_mask], bins=np.linspace(0,1,31), ax=ax, label='Correct MY', density=True)
        n = Hist(sig.max_diff[~correct_mask], bins=np.linspace(0,1,31), ax=ax, label='Incorrect MY', density=True)
        ax.set_xlabel(r'$s_\mathrm{max} - s_\mathrm{n-max}$')
        ax.set_ylabel('AU')
        ax.set_title(sig.sample)

        pdf.savefig(dpi=300)
        plt.close()



print("DONE")