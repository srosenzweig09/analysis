
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pickle import load, dump
from os import path
from logger import info, error

class InputProcessor():
    """
    This class prepares the training examples used to train the neural network.
    """
    
    def __init__(self, filename, balanced=True):
        self.inputs = np.load(filename)
        
        # Option for a balanced or imbalanced set of training examples
        # Imbalanced set may be dealt with in the future but is not
        # currently an option.
        if balanced:
            self.X = self.inputs['x']
            self.y = self.inputs['y']
            self.mjj = self.inputs['mjj'] # GeV

        else:
            raise Exception("Code under construction! Only balanced training examples can be handled currently.")
        
        # Feature size of training example set
        self.param_dim = np.shape(self.X)[1]
        
        # There are more possible non-Higgs pairs in each signal event
        # These are the unused artbitrary non-Higgs pairings
        self.X_combinatorics = self.inputs['extra_bkgd_x']
        self.y_combinatorics = self.inputs['extra_bkgd_y']
        self.mjj_combinatorics = self.inputs['extra_bkgd_mjj'] # GeV

    
    def sort_inputs(self):
        # Select Higgs pairs (labelled y=1)
        self.Higgs_training_examples = (self.y == 1)
        
        # Separate Higgs and non-Higgs examples
        self.X_Higgs = self.X[self.Higgs_training_examples]
        self.X_nonHiggs = self.X[~self.Higgs_training_examples]

        return self.X_Higgs, self.X_nonHiggs

    def normalize_inputs(self, filepath=None, save_loc=None):
        
        # If the user provides a path to a scaler file, it will be used
        # If not, a new scaler will be created and saved.
        try:
            self.xscaler = load(open(filepath + 'scaler.pkl', 'rb'))
            info(f"Scaler loaded:\n        {filepath + 'scaler.pkl'}")
        except TypeError:
            self.xscaler = MinMaxScaler()
            self.xscaler.fit(self.X)
            if not path.exists(save_loc + 'scaler.pkl'):
                dump(self.xscaler, open(save_loc + 'scaler.pkl', 'wb'))
                info(f"Scaler saved to:\n       {save_loc + 'scaler.pkl'}")
            
        self.xnormalized = self.xscaler.transform(self.X)
        
    def split_input_examples(self, test_size = 0.20, val_size = 0.125, save_loc=None, run_code=None):
        
        # Separate out the test set by selecting test_size*100% of the training examples
        X_train, X_test, x_train, x_test, y_train, y_test, _, mjj_test = train_test_split(self.X, 
                                                                                          self.xnormalized,
                                                                                          self.y, 
                                                                                          self.mjj,
                                                                                          test_size=test_size)
        
        # Separate out the validation set by selecting val_size*100% of the training examples
        X_train, X_val, x_train, x_val, y_train, y_val = train_test_split(X_train, 
                                                                          x_train, 
                                                                          y_train, 
                                                                          test_size=val_size)
    
        # Raw training examples
        self.X_train = X_train
        self.X_test  = X_test
        self.X_val   = X_val
        
        # Normalized training examples
        self.x_train = x_train
        self.x_test  = x_test
        self.x_val   = x_val
        
        # Training labels
        self.y_train = y_train
        self.y_test  = y_test
        self.y_val   = y_val
        
        # Dijet invariant masses for evaluation
        self.mjj_test  = mjj_test
        
        # Save the test sets for later evaluation
        np.savez(save_loc + 'test_set_' + run_code + '.npz', x_test=self.x_test, y_test=self.y_test, X_test=self.X_test, mjj_test=self.mjj_test)

        info("Saving test sets.")
        
    def get_x(self):
        return self.x_train, self.x_test, self.x_val
    
    def get_y(self):
        return self.y_train, self.y_test, self.y_val
    
