"""
This script was originally run in JNB and hasn't been tested as a script yet.
"""

import subprocess, shlex
import uproot

def get_branches(fname, presel_base, maxbtag_base):
    tree = uproot.lazy(f"{presel_base}/{fname}:sixBtree")
    # print(f"og tree: {tree.jet_btag}")

    branches = {}
    for field in fields:
        branches[field] = tree[field]

    # branches = {
    #     # field:sixb_tree[field]
    #     field:tree[field] for field in fields
    # }

    tmp_output = f'{maxbtag_base}/{fname}'
    if 'qcd' in presel_base: tmp_output += 'ntuple.root'
    # tmp_output = f'ntuple.root'
    try:
        with uproot.recreate(tmp_output) as f:
            # for key, value in kwargs.items():
                # f[key] = value
            f['sixBtree']=branches
            # print("f.mktree()")
            # f.mktree('sixBtree', types)
            # print("f.extend(tree)")
            # f['sixBtree'].extend(masked_tree)
    except ValueError:
        ...

qcd_presel_base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/cutflow_studies/presel/QCD'
qcd_maxbtag_base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/QCD'

ttbar_presel_base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/cutflow_studies/presel/TTJets/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8'
ttbar_maxbtag_base = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/TTJets/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8'

qcd_file_list = subprocess.check_output(shlex.split(f'ls {qcd_presel_base}')).decode('UTF-8').split('\n')
ttbar_file_list = subprocess.check_output(shlex.split(f'ls {ttbar_presel_base}')).decode('UTF-8').split('\n')

fields = []
for field in tree.fields:
    # if field.startswith('jet'): 
        # fields.append(field)
    if field.startswith('HX'): fields.append(field)
    if field.startswith('H1'): fields.append(field)
    if field.startswith('H2'): fields.append(field)
    if field.startswith('X'): fields.append(field)
    if field.startswith('Y'): fields.append(field)

jet_list = ['jet_btag', 'jet_pt', 'jet_ptRegressed', 'jet_eta', 'jet_phi', 'jet_m', 'jet_mRegressed']
for jet_var in jet_list: fields.append(jet_var)

for fname in qcd_file_list:
    print(f"processing {fname}")
    get_branches(fname)

for fname in ttbar_file_list:
    print(f"processing {fname}")
    get_branches(fname)


tree = uproot.lazy('/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/cutflow_studies/presel/NMSSM/NMSSM_XYH_YToHH_6b_MX_700_MY_400_2M/ntuple.root:sixBtree')
tmp_output = 'testing.root'

try:
    with uproot.recreate(tmp_output) as f:
        # for key, value in kwargs.items():
            # f[key] = value
        f['sixBtree']=branches
        # print("f.mktree()")
        # f.mktree('sixBtree', types)
        # print("f.extend(tree)")
        # f['sixBtree'].extend(masked_tree)
except ValueError:
    ...