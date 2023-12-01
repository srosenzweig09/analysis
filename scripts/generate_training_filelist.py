import subprocess, shlex
import sys

filepath = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM'
cmd = f"ls {filepath}"
output = subprocess.check_output(shlex.split(cmd))
output = output.decode('utf-8')
masslist = [out.removeprefix('NMSSM_XYH_YToHH_6b_') for out in output.split('\n') if out.startswith('NMSSM_XYH_YToHH_6b_')]
print(masslist)
print(len(masslist))
# masslist = [f"{filepath}/{out}/ntuple.root" for out in masslist]

# sys.exit()

# filepath = '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM/NMSSM_XYH_YToHH_6b_'

# masslist = [
#     'MX_400_MY_250_10M',
    
#     # 'MX_450_MY_250_5M',
#     'MX_450_MY_300_10M',
    
#     'MX_500_MY_250_10M',
#     'MX_500_MY_350_500k',
    
#     'MX_550_MY_250_500k',
#     'MX_550_MY_300_4M',
#     'MX_550_MY_350_500k',
#     'MX_550_MY_400_500k',

#     'MX_600_MY_250_4M',
#     'MX_600_MY_300',
#     'MX_600_MY_350_2M',
#     'MX_600_MY_400',
#     'MX_600_MY_450_500k',

#     'MX_650_MY_250_500k',
#     'MX_650_MY_300_500k',
#     'MX_650_MY_350_500k',
#     'MX_650_MY_400_500k',
#     'MX_650_MY_450_500k',
#     'MX_650_MY_500_500k',

#     'MX_700_MY_250_2M',
#     # 'MX_700_MY_300',
#     'MX_700_MY_350',
#     'MX_700_MY_400_2M',
#     'MX_700_MY_450',
#     # 'MX_700_MY_500',

#     'MX_750_MY_250',
#     'MX_750_MY_300',
#     'MX_750_MY_350',
#     'MX_750_MY_400',
#     'MX_750_MY_450',
#     'MX_750_MY_500',
#     # 'MX_750_MY_600',

#     'MX_800_MY_250_2M',
#     'MX_800_MY_300',
#     'MX_800_MY_350_2M',
#     'MX_800_MY_400',
#     # 'MX_800_MY_450',
#     'MX_800_MY_500_2M',
#     'MX_800_MY_600_2M',
 
#     # 'MX_850_MY_250',
#     # 'MX_850_MY_300',
#     # 'MX_850_MY_350',
#     # 'MX_850_MY_400',
#     # 'MX_850_MY_450',
#     # 'MX_850_MY_500',
#     # 'MX_850_MY_600',
#     # 'MX_850_MY_700',

#     'MX_900_MY_250_2M',
#     'MX_900_MY_300_2M',
#     # 'MX_900_MY_350',
#     'MX_900_MY_400',
#     # 'MX_900_MY_450',
#     'MX_900_MY_500_2M',
#     'MX_900_MY_600_2M',
#     'MX_900_MY_700_2M',

#     # 'MX_950_MY_250',
#     # 'MX_950_MY_300',
#     # 'MX_950_MY_350',
#     # 'MX_950_MY_400',
#     # 'MX_950_MY_450',
#     # 'MX_950_MY_500',
#     # 'MX_950_MY_600',
#     # 'MX_950_MY_700',
#     # 'MX_950_MY_800',

#     'MX_1000_MY_250_2M',
#     'MX_1000_MY_300',
#     # 'MX_1000_MY_350',
#     'MX_1000_MY_400_2M',
#     'MX_1000_MY_450_2M',
#     'MX_1000_MY_500_2M',
#     'MX_1000_MY_600_3M',
#     'MX_1000_MY_700_2M',
#     'MX_1000_MY_800_2M',

#     'MX_1100_MY_250_2M',
#     'MX_1100_MY_300_2M',
#     # 'MX_1100_MY_350',
#     'MX_1100_MY_400',
#     'MX_1100_MY_450_2M',
#     'MX_1100_MY_500_2M',
#     'MX_1100_MY_600_2M',
#     'MX_1100_MY_700_2M',
#     'MX_1100_MY_800_2M',
#     'MX_1100_MY_900_2M',

#     'MX_1200_MY_250_2M',
#     'MX_1200_MY_300',
#     # 'MX_1200_MY_350',
#     'MX_1200_MY_400_2M',
#     'MX_1200_MY_450_2M',
#     'MX_1200_MY_500_2M',
#     'MX_1200_MY_600_2M',
#     'MX_1200_MY_700_2M',
#     'MX_1200_MY_800_2M',
#     'MX_1200_MY_900_2M',
#     'MX_1200_MY_1000_2M',
#     ]

print(','.join(masslist))
print("MX =", [int(v.split('_')[1]) for v in masslist])
print("MY =", [int(v.split('_')[3]) for v in masslist])

# ntuple_filelist = [f"{filepath}{file}/ntuple.root\n" for file in masslist]
ntuple_filelist = [f"{filepath}/NMSSM_XYH_YToHH_6b_{file}/ntuple.root\n" for file in masslist]
# fully_res_filelist = [f"{filepath}{file}/fully_res_ntuple.root\n" for file in masslist]
fully_res_filelist = [f"{filepath}/NMSSM_XYH_YToHH_6b_{file}/fully_res_ntuple.root\n" for file in masslist]

ntuple_filelist += [
    '/eos/uscms/store/user/srosenzw/sixb/ntuples/Autumn18/maxbtag/QCD/*/ntuple.root\n',
    '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/TTJets/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/ntuple.root\n',
]
fully_res_filelist += [
    '/eos/uscms/store/user/srosenzw/sixb/ntuples/Autumn18/maxbtag/QCD/*/training_ntuple.root\n',
    '/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/TTJets/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/training_ntuple.root\n',
]

print(len(ntuple_filelist))

f_ntuple_filelist = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/eightbStudies/scripts/ntuple_filelist.txt'
with open(f_ntuple_filelist, "w") as f:
    f.writelines(ntuple_filelist)

f_fully_res_filelist = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/eightbStudies/scripts/fully_res_filelist.txt'
with open(f_fully_res_filelist, "w") as f:
    f.writelines(fully_res_filelist)