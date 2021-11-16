

# data cards
DataCards_dir = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/data/sixb/'

def buildDataCard(S, B, uncB=0):
    DataCardText = [
    'imax 1  number of channels\n',
    'jmax 1  number of backgrounds\n',
    'kmax *  number of nuisance parameters\n',
    '-------\n',
    'bin bin1\n',
    'observation 0\n',
    '-------\n',
    'bin          bin1     bin1\n',
    'process      sig      bkg\n',
    'process      0        1 \n',
    f'rate         {S}     {B}\n',
    '-------\n',
    '\n',
    'lumi   lnN   1.01      -\n',
    f'other  lnN   -         {1+uncB}\n'
    ]
    return DataCardText