"""The goal of this script is to determine the best variables to use for BDT training by performing an iterative removal algorithm
"""

print("---------------------------------")
print("STARTING PROGRAM")
print("---------------------------------")

from argparse import ArgumentParser
from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import sys
from utils.analysis import Signal
from utils.plotter import Hist, Ratio
import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

### ------------------------------------------------------------------------------------
## Implement command line parser

print(".. parsing command line arguments")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--cfg', dest='cfg', help='config file', default='')

# region shapes
parser.add_argument('-r', '--rectangular', dest='rectangular', help='', action='store_true', default=True)
parser.add_argument('-s', '--spherical', dest='spherical', help='', action='store_true', default=False)
parser.add_argument('-n', '--no-iteration', dest='no_iteration', action='store_true', help='Only train BDT with all variables', default=False)

args = parser.parse_args()

if bool(args.rectangular): args.cfg = 'config/rectConfig.cfg'
if bool(args.spherical): args.cfg = 'config/sphereConfig.cfg'

### ------------------------------------------------------------------------------------
## Implement config parser

print(".. parsing config file")

config = ConfigParser()
config.optionxform = str
config.read(args.cfg)

base = config['file']['base']
signal = config['file']['signal']
data = config['file']['data']
treename = config['file']['tree']
year = int(config['file']['year'])
pairing = config['pairing']['scheme']
pairing_type = pairing.split('_')[0]

variables = config['BDT']['variables']
variables = variables.split(', ')

nbins = int(config['plot']['nbins'])
minMX = int(config['plot']['minMX'])
maxMX = int(config['plot']['maxMX'])
mBins = np.linspace(minMX, maxMX, nbins)

if args.spherical and args.rectangular: args.rectangular = False
if args.rectangular: region_type = 'rect'
elif args.spherical: region_type = 'sphere'

score = float(config['score']['threshold'])

indir = f"root://cmseos.fnal.gov/{base}/{pairing}/"

### ------------------------------------------------------------------------------------
## Obtain data regions

datFileName = f"{indir}{data}"
datTree = Signal(datFileName)

if args.rectangular:
    print("\n ---RECTANGULAR---")
    datTree.rectangular_region(config)
elif args.spherical:
    print("\n---SPHERICAL---")
    datTree.spherical_region(config)
else:
    raise AttributeError("No mass region definition!")

def training(variables,return_model=False):
   print(".. creating feature matrix")
   x_CRls = np.array(())
   x_CRhs = np.array(())

   for var in variables:
      weighted_model = datTree.np(var)[datTree.CRls_mask]
      target_arr = datTree.np(var)[datTree.CRhs_mask]
      x_CRls = np.append(x_CRls, weighted_model)
      x_CRhs = np.append(x_CRhs, target_arr)
   x_CRls = x_CRls.reshape(len(variables),sum(datTree.CRls_mask)).transpose()
   x_CRhs = x_CRhs.reshape(len(variables),sum(datTree.CRhs_mask)).transpose()

   data = np.row_stack((x_CRls, x_CRhs))
   labels = np.concatenate((np.ones(len(x_CRls)),np.zeros(len(x_CRhs))))
   weights = np.concatenate((datTree.CR_weights, np.ones(len(x_CRhs))))

   shape = x_CRls.shape
   temp_x_CRls = np.multiply(np.random.rand(shape[0],shape[1]), x_CRls)
   test_data = np.row_stack((temp_x_CRls, x_CRhs))
   test_weights = datTree.CR_weights * np.random.random_sample(len(datTree.CR_weights))
   test_weights = np.append(test_weights, np.ones(len(x_CRhs)))

   # division into homogeneous subgroups
   cross_val = StratifiedKFold(n_splits=4, shuffle=True, random_state=2020)

   tprs = []
   aucs = []
   acc_scores = []
   mean_fpr = np.linspace(0, 1, 100)
   for i,(train, test) in enumerate(cross_val.split(data, labels, weights)):
      classifier = GradientBoostingClassifier(n_estimators=200,subsample=0.62,random_state=2020)
      model = classifier.fit(data[train], labels[train],sample_weight=weights[train])
      proba = model.predict_proba(data[test])
      test_proba = model.predict_proba(test_data[test])
      fpr, tpr, thresholds = roc_curve(labels[test], proba[:, 1], sample_weight=weights[test])
      pred = classifier.predict(data[test])
      test_pred = classifier.predict(test_data[test])
      acc_score = accuracy_score(labels[test], pred, sample_weight=weights[test])
      tprs.append(np.interp(mean_fpr, fpr, tpr))
      tprs[-1][0] = 0.0
      roc_auc = auc(fpr, tpr)
      aucs.append(roc_auc)
      acc_scores.append(acc_score)
      # print(f"    ** Mean ROC AUC (fold {i}) = {roc_auc:.3f}")
   print(f"    ** Variable dropped: {var}")
   mean_auc = round(np.asarray(aucs).mean(),3)
   mean_acc = round(np.asarray(acc_scores).mean(),3)
   print(f"Mean AUC = {mean_auc}")
   print(f"Mean acc = {mean_acc}")
   print()
   print('----------------------------------------')
   print()
   if return_model: return mean_auc, mean_acc, classifier
   return mean_auc, mean_acc

print(".. WITH ALL VARIABLES PRESENT")
VR_weights, SR_weights = datTree.bdt_process(region_type, config)
mean_auc, mean_acc, model = training(variables, return_model=True)

fig, axs = plt.subplots(nrows=7, ncols=3, figsize=(21,9))
fig.suptitle('Control Region')
for ax,var in zip(axs, variables):
   Hist(abs(datTree.get(var)[datTree.CRhs_mask]), bins=mBins, ax=ax, label='Target')
   # Hist(abs(datTree.get(var)[datTree.CRls_mask]), bins=mBins, ax=ax, label='Original')
   Hist(abs(datTree.get(var)[datTree.CRls_mask]), bins=mBins, weights=datTree.CR_weights, ax=ax, label='Model')
   ax.set_xlabel(var)
   ax.set_ylabel('Events')
   ax.legend()

fig, axs = plt.subplots(nrows=7, ncols=3, figsize=(21,9))
fig.suptitle('Validation Region')
for ax,var in zip(axs, variables):
   Hist(abs(datTree.get(var)[datTree.VRhs_mask]), bins=mBins, ax=ax, label='Target')
   # Hist(abs(datTree.get(var)[datTree.CRls_mask]), bins=mBins, ax=ax, label='Original')
   Hist(abs(datTree.get(var)[datTree.VRls_mask]), bins=mBins, weights=datTree.VR_weights, ax=ax, label='Model')
   ax.set_xlabel(var)
   ax.set_ylabel('Events')
   ax.legend()

if args.no_iteration: sys.exit()

var_auc = []
var_acc = []
for var in variables:
   print(var)
   temp_vars = np.asarray(variables)
   var_mask = temp_vars == var
   temp_vars = temp_vars[~var_mask]
   config['BDT']['variables'] = ', '.join(temp_vars.tolist())
   print(config['BDT']['variables'])
   # CR(data/model), VR (model), and SR (prediction)
   VR_weights, SR_weights = datTree.bdt_process(region_type, config)
   mean_auc, mean_acc = training(temp_vars)
   var_auc.append(mean_auc)
   var_acc.append(mean_acc)

var_auc = np.asarray(var_auc)
var_acc = np.asarray(var_acc)

N = len(variables)
x = range(N)
y_auc = [mean_auc]*N
y_acc = [mean_acc]*N

auc_dist = np.around(var_auc - mean_auc, 3)
acc_dist = np.around(var_acc - mean_acc, 3)
tot_dist = abs(auc_dist) + abs(acc_dist)

min_ind = np.argmin(tot_dist)
sorted_ind = np.argsort(tot_dist)
sorted_tot_dist = tot_dist[sorted_ind]
sorted_auc_dist = np.asarray(auc_dist)[sorted_ind]
sorted_acc_dist = np.asarray(acc_dist)[sorted_ind]
var_auc = var_auc[sorted_ind]
var_acc = var_acc[sorted_ind]
sorted_vars     = np.asarray(variables, dtype=str)[sorted_ind]

# fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10,10), sharex=True, gridspec_kw={'hspace':0.05})
# ax = axs[0]
# ax.plot(sorted_vars, y_auc, '--', color='C0')
# ax.plot(sorted_vars, y_acc, '--', color='C1')
# for i,(auc,acc) in enumerate(zip(var_auc,var_acc)):
#    ax.plot([i,i], [0.5,auc], color='C0')
#    ax.plot([i,i], [0.5,acc], color='C1')
# ax.scatter(sorted_vars, var_auc, label='auc', color='C0')
# ax.scatter(sorted_vars, var_acc, label='accuracy', color='C1')
# # ax.set_xticklabels(labels=sorted_vars, fontdict={'rotation':45,'fontsize':10,'ha':'right'})
# ax.tick_params(axis='x', labelbottom=False)
# ax.legend(fontsize=12)

# ax = axs[1]
# ax.scatter(sorted_vars, sorted_tot_dist, label='sorted distance')
# ax.set_xticklabels(labels=sorted_vars, fontdict={'rotation':45,'fontsize':12,'ha':'right'})
# ax.set_ylabel("Total Distance")

# for ax in axs:
#    ax.ticklabel_format(axis='y', style='plain')
# plt.subplots_adjust(bottom=0.2)

# plt.show()
# # fig.savefig('plots/bdt_performance/iter_vars_auc_acc.pdf', bbox_inches='tight')