import sys

# CHECK THAT THESE ARE UP TO DATE
# CHECKED AS OF 
# May 2, 2022
sphere_dict = {
   'scheme' : 'sphere',
   'jets' : 'bias',
   'pairs' : 'dHHH',
   'bkg_crtf' : 1.0,
   'bkg_vrpred' : 1.01,
   'bkg_vr_normval' : 1.01,
}
class DataCard:
   def __init__(self, signal, MCstats=True, **kwargs):
      for key in kwargs.keys():
         if key in sphere_dict.keys():
            sphere_dict[key] = kwargs[key]
      
      pairs = sphere_dict['pairs']
      jets = sphere_dict['jets']
      scheme = sphere_dict['scheme']
      error = sphere_dict['bkg_crtf']
      std = sphere_dict['bkg_vrpred']
      k = sphere_dict['bkg_vr_normval']
 
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
      ]
      if MCstats: dataCardText.append('* autoMCStats 10\n')
      # if no_bkg_stats: sigROOT = sigROOT + '_nobkgstats'
      outfile = f"combine/{pairs}/{jets}/{scheme}/datacards/{signal}.txt"
      print(outfile)

      with open(outfile, "w") as f:
         f.writelines(dataCardText)
      

# def createDataCard(location, sigROOT, h_name, err_dict, outdir, no_bkg_stats=True, MCstats=True):
#     """Creates datacard for a given tree."""
#     bkg_crtf = err_dict['bkg_crtf']
#     bkg_vrpred = err_dict['bkg_vrpred']
#     bkg_vr_normval = err_dict['bkg_vr_normval']

#     dataCardText = [
#         'imax 1  number of channels\n',
#         'jmax 1  number of backgrounds\n',
#         'kmax *  number of nuisance parameters\n',
#         '-------\n',
#         f'shapes sig bin1 {location}/{sigROOT}.root {h_name}\n',
#         f'shapes bkg bin1 {location}/data.root h_dat\n',
#         f'shapes data_obs bin1 {location}/data.root h_dat\n',
#         '_______\n',
#         'bin bin1\n',
#         'observation -1\n',
#         '-------\n',
#         'bin          bin1     bin1\n',
#         'process      sig      bkg\n',
#         'process      0        1 \n',
#         'rate         -1       -1\n',
#         '-------\n',
#         '\n',
#         f'error  lnN   -        {bkg_crtf}\n',
#         f'sd     lnN   -        {bkg_vrpred}\n',
#         f'k      lnN   -        {bkg_vr_normval}\n'
#     ]
#     if MCstats: dataCardText.append('* autoMCStats 10\n')
#     if no_bkg_stats: sigROOT = sigROOT + '_nobkgstats'
#     print(f"{outdir}/{sigROOT}.txt")

#     with open(f"{outdir}/{sigROOT}.txt", "w") as f:
#         f.writelines(dataCardText)
