import uproot3
import awkward0
from logger import info
import numpy as np

def get_uproot_Table(filename, tree_name='sixBtree'):
    
    f = uproot3.open(filename)
        
    tree     = f[tree_name]
    branches = tree.arrays(namedecode='utf-8')
    table    = awkward0.Table(branches)

    ncols = 3
    keys = table.columns
    n = len(keys)
    modu = n%ncols
    
    print("-"*100)
    print(" "*44 + "TABLE COLUMNS" + " "*43)
    print("-"*100)
    for i in np.arange(0,n-modu,ncols):
        row = ""
        for j in np.arange(i,i+ncols):
            row += "{:<34}".format(keys[j])
        print(row)
    if modu != 0:
        row = ""
        for i in np.arange(n-modu, n):
            row += "{:<34}".format(keys[i])
        print(row)
    print("-"*100)

    return table