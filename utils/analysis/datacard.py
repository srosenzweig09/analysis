import sys
from utils.analysis.feyn import model_name
import json
from pathlib import Path

sixb_dir = "/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb"
HiggsCombine_dir = "/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit"

HC_path = f"{HiggsCombine_dir}/datacards/feynnet/{model_name}"
sixb_path = f"{sixb_dir}/combine/feynnet/{model_name}/datacards"

json_path = f"{sixb_dir}/combine/feynnet/{model_name}/model.json"
with open(json_path) as f: 
   data = json.load(f)

lumi_dict = {
   2016 : 1.012,
   2017 : 1.023,
   2018 : 1.025
}

class DataCard:

   def __init__(self, signal, year=2018):
      self.signal = signal

      error = data['crtf']
      std = data['vr_stat_prec']
      k = data['vr_yield_val']
      norm = data['norm']
      lumi = lumi_dict[year]
 
      self.dataCardHeader = [
         'imax 1  number of channels\n',
         'jmax 1  number of backgrounds\n',
         'kmax *  number of nuisance parameters\n',
         '-------\n',
         f'shapes signal * {sixb_dir}/combine/feynnet/{model_name}/{signal}.root $PROCESS $PROCESS_$SYSTEMATIC\n',
         f'shapes model bin1 {sixb_dir}/combine/feynnet/{model_name}/model.root $PROCESS $PROCESS_$SYSTEMATIC\n',
         f'shapes data_obs bin1 {sixb_dir}/combine/feynnet/{model_name}/model.root model\n',
         '_______\n',
         'bin bin1\n',
         'observation -1\n',
         '-------\n',
         'bin          bin1     bin1\n',
         'process      signal   model\n',
         'process      0        1 \n',
         'rate         -1       -1\n',
         '-------\n',
      ]

      self.dataCardSystematics = [
         # f'error                   lnN         -       {error}\n',
         # f'sd                      lnN         -       {std}\n',
         # f'k                       lnN         -       {k}\n',
         f'Norm                    lnN         -       {norm}\n',
         f'Lumi                    lnN       {lumi}     -\n',
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
         'bJER                    shape       1       -\n',
         'JERpt                   shape       1       -\n',
         # 'JEReta                  shape       1       -\n',
         # 'JERphi                  shape       1       -\n',
         'Trigger                 shape       1       -\n',
         'Pileup                  shape       1       -\n',
         'PUID                    shape       1       -\n',
         'BTagHF                  shape       1       -\n',
         'BTagLF                  shape       1       -\n',
         'BTagHFStats1            shape       1       -\n',
         'BTagHFStats2            shape       1       -\n',
         'BTagLFStats1            shape       1       -\n',
         'BTagLFStats2            shape       1       -\n',
         'CRShift                 shape       -       1\n',
         'AvgBTag                 shape       -       1\n',
         '* autoMCStats 10\n'
      ]

      self.dataCardText = self.dataCardHeader + self.dataCardSystematics

      # self.dataCardText.append('* autoMCStats 10\n')
      # if no_bkg_stats: sigROOT = sigROOT + '_nobkgstats'

   def add_systematic(self, syst, stype, sample):
      """
      syst is the systematic name
      stype is either lnN or shape
      sample is either signal or model
      """
      if sample == 'signal': suffix = '1       -'
      elif sample == 'model': suffix = '-       1'
      else: raise ValueError(f"Unknown sample: {sample}")

      line = f'{syst.ljust(24)}{stype.ljust(12)}{suffix}\n'

   def print(self):
      return self.dataCardText

   def write_no_systematics(self):
      dirname = "no_systematics"

      Path(f"{sixb_path}/{dirname}").mkdir(parents=True, exist_ok=True)
      Path(f"{HC_path}/{dirname}").mkdir(parents=True, exist_ok=True)

      sixb_outfile = f"{sixb_path}/{dirname}/{self.signal}.txt"
      HC_outfile = f"{HC_path}/{dirname}/{self.signal}.txt"

      with open(sixb_outfile, "w") as f:
         f.writelines(self.dataCardHeader)

      with open(HC_outfile, "w") as f:
         f.writelines(self.dataCardHeader)
   
   def write_additions(self):
      for line in self.dataCardSystematics:
         syst = line.split()[0]
         tmp = line

         if syst == "*": syst = 'mcstats'
         dirname = "only_" + syst
         Path(f"{sixb_path}/{dirname}").mkdir(parents=True, exist_ok=True)
         Path(f"{HC_path}/{dirname}").mkdir(parents=True, exist_ok=True)

         with open(f"{sixb_path}/{dirname}/{self.signal}.txt", "w") as f:
            f.writelines(self.dataCardHeader + [tmp])
         
         with open(f"{HC_path}/{dirname}/{self.signal}.txt", "w") as f:
            f.writelines(self.dataCardHeader + [tmp])

   def write_removals(self):
      for i,line in enumerate(self.dataCardSystematics):
         syst = line.split()[0]
         tmp = self.dataCardSystematics.copy()
         tmp.pop(i)

         if syst == "*": syst = 'mcstats'
         dirname = "all_minus_" + syst
         Path(f"{sixb_path}/{dirname}").mkdir(parents=True, exist_ok=True)
         Path(f"{HC_path}/{dirname}").mkdir(parents=True, exist_ok=True)

         with open(f"{sixb_path}/{dirname}/{self.signal}.txt", "w") as f:
            f.writelines(self.dataCardHeader + tmp)
         
         with open(f"{HC_path}/{dirname}/{self.signal}.txt", "w") as f:
            f.writelines(self.dataCardHeader + tmp)


   def write(self):
      Path(sixb_path).mkdir(parents=True, exist_ok=True)
      Path(HC_path).mkdir(parents=True, exist_ok=True)

      sixb_outfile = f"{sixb_path}/{self.signal}.txt"
      HC_outfile = f"{HC_path}/{self.signal}.txt"

      with open(sixb_outfile, "w") as f:
         f.writelines(self.dataCardText)

      with open(HC_outfile, "w") as f:
         f.writelines(self.dataCardText)

      