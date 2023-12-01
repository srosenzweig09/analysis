import sys

class DataCard:
   # def __init__(self, signal, MCstats=True, **kwargs):

   def __init__(self, data):
      """
      data : dictionary containing crtf, vr_stat_prec, and vr_yield_val
      """

      error = data['crtf']
      std = data['vr_stat_prec']
      k = data['vr_yield_val']
 
      dataCardText = [
         'imax 1  number of channels\n',
         'jmax 1  number of backgrounds\n',
         'kmax *  number of nuisance parameters\n',
         '-------\n',
         f'shapes signal * /uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/combine/{pairs}/{jets}/{scheme}/root/{signal}.root $PROCESS $PROCESS_$SYSTEMATIC\n',
         f'shapes bkg bin1 /uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/combine/dHHH/sphere/data.root data\n',
         f'shapes data_obs bin1 /uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/combine/dHHH/sphere/data.root data\n',
         '_______\n',
         'bin bin1\n',
         'observation -1\n',
         '-------\n',
         'bin          bin1     bin1\n',
         'process      signal   bkg\n',
         'process      0        1 \n',
         'rate         -1       -1\n',
         '-------\n',
         f'error                  lnN         -       {error}\n',
         f'sd                     lnN         -       {std}\n',
         f'k                      lnN         -       {k}\n',
         'lumi                    lnN       1.025     -\n',
         'Absolute                shape       1       -\n',
         'Absolute_2018           shape       1       -\n',
         'BBEC1                   shape       1       -\n',
         'BBEC1_2018              shape       1       -\n',
         'EC2                     shape       1       -\n',
         'EC2_2018                shape       1       -\n',
         'FlavorQCD               shape       1       -\n',
         'HF                      shape       1       -\n',
         'HF_2018                 shape       1       -\n',
         'RelativeBal             shape       1       -\n',
         'RelativeSample_2018     shape       1       -\n',
         'JER_pt                  shape       1       -\n',
         'JER_eta                 shape       1       -\n',
         'JER_phi                 shape       1       -\n',
         'Trigger                 shape       1       -\n',
         'Pileup                  shape       1       -\n',
         'PUID                    shape       1       -\n',
         'BTagHF                  shape       1       -\n',
         'BTagLF                  shape       1       -\n',
         'BTagHFStats1            shape       1       -\n',
         'BTagHFStats2            shape       1       -\n',
         'BTagLFStats1            shape       1       -\n',
         'BTagLFStats2            shape       1       -\n',

      ]
      if MCstats: dataCardText.append('* autoMCStats 10\n')
      # if no_bkg_stats: sigROOT = sigROOT + '_nobkgstats'
      outfile = f"combine/{pairs}/{jets}/{scheme}/datacards/{signal}.txt"
      print(outfile)

      with open(outfile, "w") as f:
         f.writelines(dataCardText)
      