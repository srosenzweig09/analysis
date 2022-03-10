import awkward as ak
from hep_ml import reweight
import numpy as np
import pandas as pd
import sys

class Trainer():
    def __init__(self, config):
        self.Nestimators  = int(config['BDT']['Nestimators'])
        self.learningRate = float(config['BDT']['learningRate'])
        self.maxDepth     = int(config['BDT']['maxDepth'])
        self.minLeaves    = int(config['BDT']['minLeaves'])
        self.GBsubsample  = float(config['BDT']['GBsubsample'])
        self.randomState  = int(config['BDT']['randomState'])
        self.variables    = config['BDT']['variables'].split(", ")

    def get_df(self, mask, variables):
        features = {}
        for var in self.variables:
            features[var] = abs(self.tree.get(var)[mask])
        df = pd.DataFrame(features)
        return df

    def train_bdt(self, ls_mask, hs_mask):

        # original_mask = ls_mask
        TF = sum(hs_mask)/sum(ls_mask)
        ls_weights = np.ones(ak.sum(ls_mask))*TF
        hs_weights = np.ones(ak.sum([hs_mask]))

        df_ls = self.get_df(ls_mask, self.variables)
        df_hs = self.get_df(hs_mask, self.variables)

        np.random.seed(self.randomState) #Fix any random seed using numpy arrays
        print(".. calling reweight.GBReweighter")
        reweighter_base      = reweight.GBReweighter(
            n_estimators     = self.Nestimators, 
            learning_rate    = self.learningRate, 
            max_depth        = self.maxDepth, 
            min_samples_leaf = self.minLeaves,
            gb_args          = {'subsample': self.GBsubsample})

        print(".. calling reweight.FoldingReweighter")
        reweighter = reweight.FoldingReweighter(reweighter_base, random_state=self.randomState, n_folds=2, verbose=False)

        print(".. calling reweighter.fit\n")
        reweighter.fit(df_ls,df_hs,ls_weights,hs_weights)
        self.reweighter = reweighter

    def bdt_prediction(self, mask, SR=True):
        df_ls = self.get_df(mask, self.variables)
        initial_weights = np.ones(ak.sum(mask))*self.TF
        weights_pred = self.reweighter.predict_weights(df_ls,initial_weights,lambda x: np.mean(x, axis=0))
        return weights_pred

class RectTrainer(Trainer):
    def __init__(self, tree, config):
        super().__init__(config)
        self.tree = tree
        self.config = config

        self.tree.rectangular_region(config)
        
        self.train_bdt(self.tree.CRls_mask, self.tree.CRhs_mask)
        bdt_weights = self.bdt_prediction(self.CRls_mask)
        

    

class SphereTrainer(Trainer):
    def __init__(self, tree, config):
        super().__init__(config)
        self.tree = tree
        self.config = config

        self.tree.spherical_region(config)
        
        self.train_bdt(self.tree.V_CRls_mask, self.tree.V_CRhs_mask)


        self.train_bdt(self.tree.A_CRls_mask, self.tree.A_CRhs_mask)




