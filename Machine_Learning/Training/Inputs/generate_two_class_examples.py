import numpy as np
from argparse import ArgumentParser

from logger import info

### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')
parser.add_argument('--type'       , dest =  'type'      , help = 'parton, smeared, or reco'      ,  required = True    )

args = parser.parse_args()

### ------------------------------------------------------------------------------------
## 

if args.type == 'parton':
    examples = np.load('Gen_Inputs/nn_input_MX700_MY400_class.npz')
    save_loc = 'Gen_Inputs/nn_input_MX700_MY400_two_class.npz'
if args.type == 'smeared':
    examples = np.load('Gen_Inputs/nn_input_MX700_MY400_class_smeared.npz')
    save_loc = 'Gen_Inputs/nn_input_MX700_MY400_two_class_smeared.npz'
if args.type == 'reco':
    examples = np.load('Reco_Inputs/nn_input_MX700_MY400_class.npz')
    save_loc = 'Reco_Inputs/nn_input_MX700_MY400_two_class'

y = examples['y']

x = examples['x']

x = np.delete(x, axis=1, obj=6)
x = np.column_stack((x, x[:,0]**2, x[:,3]**2))

print(y)

new_y = np.array(())


for i,targets in enumerate(y):
    if i%10000 == 0: print(f"Processing example {i}/{len(y)}")
    if targets == 0:
        new_y = np.append(new_y, [1,0])
    elif targets == 1:
        new_y = np.append(new_y, [0,1])
    else:
        print("This shouldn't print.")

new_y = new_y.reshape(len(y),2)

np.savez(save_loc + '_9features', x=x,  y=new_y,  mjj=examples['mjj'], params=examples['params'])