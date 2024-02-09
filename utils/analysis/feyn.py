import json, uproot
import numpy as np
import awkward as ak

model_name = '20230731_7d266883bbfb88fe4e226783a7d1c9db_ranger_lr0.0047_batch2000_withbkg'
old_model_path = f'/eos/uscms/store/user/srosenzw/weaver/models/exp_sixb_official/feynnet_ranker_6b/{model_name}/predict_output'

new_model_path = "/eos/uscms/store/user/srosenzw/weaver/cmsuf/data/store/user/srosenzw/lightning/models/feynnet_lightning/X_YH_3H_6b/x3h/lightning_logs/version_23183119/predict/"

def getMassDict(year):
    with open(f"{new_model_path}/{year}/samples.json", 'r') as file:
        mass_dict = json.load(file)
    return mass_dict

class Model():

    def __init__(self, which, sixb):
        mass_dict = getMassDict(sixb.year)

        if which == 'new': self.init_new_model(sixb, mass_dict)
        elif which == 'old': self.init_old_model(sixb)
        else: raise ValueError(f"Model type '{which}' not recognized")

    def init_new_model(self, sixb, mass_dict):
        print("------ new -------")
        hash = mass_dict[sixb.filename]
        model_path = f"{new_model_path}/{sixb.year}/{hash}.root"
        with uproot.open(model_path) as f:
            t = f['Events']
            # print(t['sorted_rank'].array())
            maxcomb = ak.firsts(t['sorted_j_assignments'].array())
        self.combos = ak.from_regular(maxcomb)
        # print(print(combos))

    def init_old_model(self, sixb):
        print("------ old -------")
        model_path = f"{old_model_path}/{sixb.year}/{sixb.filename}.root"
        with uproot.open(model_path) as f:
            f = f['Events']
            maxcomb = f['max_comb'].array(library='np')

        combos = maxcomb.astype(int)
        self.combos = ak.from_regular(combos)
