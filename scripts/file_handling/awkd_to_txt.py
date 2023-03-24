import awkward0 as ak0
import awkward as ak
import uproot as up
import re
import subprocess, shlex
import sys

model_output = 'weaver-benchmark/weaver/X_YH_3H_6b/models/rank_graph_6jets/output'

base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/studies'
cmd = f'ls {base}'
output = subprocess.run(shlex.split(cmd), capture_output=True)
files = output.stdout.decode('UTF-8').split('\n')

for f in files:
    ntuple = 'ntuple.root'
    if 'tar' in f: continue
    if '_10M' in f: continue
    if '_100k' in f: continue
    if '_2M' in f: 
        ntuple = 'ntuple_test.root'

    n_jet = up.open(f"{base}/{f}/{ntuple}:sixBtree/n_jet").array()
    fname = f[re.search('MX_', f).start():]

    with ak0.load(f"{model_output}/{fname}.awkd") as ak0array:
        scores = ak.unflatten(ak.from_awkward0(ak0array['scores']), n_jet)

    lines = []
    for line in scores.to_list():
        line = [str(score) for score in line]
        line = " ".join(line) + "\n"
        lines.append(line)

    with open(f"/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixB/analysis/sixBanalysis/scores/{fname}.txt", 'w') as out:
        out.writelines(lines)


# data_njet = up.open("/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/presel/JetHT_Data_UL/JetHT_Run2018_full/ntuple.root:sixBtree/n_jet").array()
# # data_njet = up.open("/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/studies/NMSSM_XYH_YToHH_6b_MX_700_MY_400_2M/ntuple_test.root:sixBtree/n_jet").array()

# with ak0.load('weaver-benchmark/weaver/X_YH_3H_6b/models/rank_graph_6jets/output/data.awkd') as ak0array:
#     data_scores = ak.unflatten(ak.from_awkward0(ak0array['scores']), data_njet)

# lines = []
# for line in data_scores.to_list():
#     line = [str(score) for score in line]
#     line = " ".join(line) + "\n"
#     lines.append(line)

# with open("/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixB/analysis/sixBanalysis/data_scores.txt", 'w') as f:
#     f.writelines(lines)