import uproot3
import awkward0
from logger import info

def get_uproot_Table(filename, tree_name='sixBtree'):
    
    f = uproot3.open(filename)
    print(f.keys())
        
    tree     = f[tree_name]
    branches = tree.arrays(namedecode='utf-8')
    table    = awkward0.Table(branches)
    
    return table