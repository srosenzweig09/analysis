from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pickle import load, dump
import numpy as np

class InputProcessor():
    
    def __init__(self, filename, balanced=True, include_dR=False):
        self.inputs = np.load(filename)
        
        if balanced:
            self.X = self.inputs['x']
            self.y = self.inputs['y']
            
            if not include_dR:
                self.X = self.X[:,:-1] # Remove DeltaR as an input parameter
        
        else:
            raise Exception("Code under construction! Only balanced training examples can be handled currently.")
            
        self.param_dim = np.shape(self.X)[1]
        
        self.X_combinatorics = self.inputs['extra_bkgd_x']
        self.y_combinatorics = self.inputs['extra_bkgd_y']
        
        self.sort_inputs()
        self.sort_mass()
    
    def sort_inputs(self):
        self.Higgs_training_examples = (self.y == 1)
        
        self.X_Higgs = self.X[self.Higgs_training_examples]
        self.X_nonHiggs = self.X[~self.Higgs_training_examples]
        
    def sort_mass(self):
        self.mjj = self.inputs['mjj'] # GeV
        self.mjj_combinatorics = self.inputs['extra_bkgd_mjj'] # GeV
        
    def normalize_inputs(self, filepath=None, save_loc=None, run_code=None):
        
        try:
            self.xscaler = load(open(filepath + 'scaler.pkl', 'rb'))
            print(f"[INFO] Scaler loaded:\n        {filepath + 'scaler.pkl'}")
        except TypeError:
            self.xscaler = MinMaxScaler()
            self.xscaler.fit(self.X)
            dump(self.xscaler, open(save_loc + 'scaler_' + run_code + '.pkl', 'wb'))
            print(f"[INFO] Scaler saved to:\n       {save_loc + 'scaler_' + run_code + '.pkl'}")
            
        self.xnormalized = self.xscaler.transform(self.X)
        
    def split_input_examples(self, test_size = 0.20, val_size = 0.125):
        
        X_train, X_test, x_train, x_test, y_train, y_test, _, mjj_test = train_test_split(self.X, self.xnormalized, self.y, self.mjj, test_size=test_size, random_state=42)
        X_train, X_val, x_train, x_val, y_train, y_val = train_test_split(X_train, x_train, y_train, test_size=val_size, random_state=42)
    
        self.X_train = X_train
        self.X_test  = X_test
        self.X_val   = X_val
        
        self.x_train = x_train
        self.x_test  = x_test
        self.x_val   = x_val
        
        self.y_train = y_train
        self.y_test  = y_test
        self.y_val   = y_val
        
        self.mjj_test  = mjj_test
        
    def get_x(self):
        return self.x_train, self.x_test, self.x_val
    
    def get_y(self):
        return self.y_train, self.y_test, self.y_val
    
