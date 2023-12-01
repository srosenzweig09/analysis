from . import *

lumiMap = {
    None:[1,None],
    2016:[35900,"(13 TeV,2016)"],
    2017:[41500,"(13 TeV,2017)"],
    2018:[59740,"(13 TeV,2018)"],
    20180:[14300,"(13 TeV,2018 A)"],
    20181:[7070,"(13 TeV,2018 B)"],
    20182:[6900,"(13 TeV,2018 C)"],
    20183:[13540,"(13 TeV,2018 D)"],
    "Run2":[101000,"13 TeV,Run 2)"],
}

tagMap = {
    "QCD":"QCD",
    "NMSSM":"Signal",
    "TTJets":"TTJets",
    "Data":"Data",
}

colorMap = {
    "QCD":"tab:blue",
    "Signal":"tab:orange",
    "TTJets":"tab:green",
    "Data":"black",
    "Bkg":"grey"
}

xsecMap = {
    "JetHT_Run2018A":'N/A',
    "JetHT_Run2018B":'N/A',
    "JetHT_Run2018C":'N/A',
    "JetHT_Run2018D":'N/A',
    
    "NMSSM": 1.0 * (5.824E-01)**3, # pb,
    "NMSSM_XYH_YToHH_8b": 1.0 * (5.824E-01)**4, # pb
    
    "QCD_Pt_15to30"    :1246000000.0,
    "QCD_Pt_30to50"    :106500000.0,
    "QCD_Pt_50to80"    :15700000.0,
    "QCD_Pt_80to120"   :2346000.0,
    "QCD_Pt_120to170"  :407300.0,
    "QCD_Pt_170to300"  :103500.0,
    "QCD_Pt_300to470"  :6826.0,
    "QCD_Pt_470to600"  :552.1,
    "QCD_Pt_600to800"  :156.5,
    "QCD_Pt_800to1000" :26.28,
    "QCD_Pt_1000to1400":7.465,
    "QCD_Pt_1400to1800":0.6484,
    "QCD_Pt_1800to2400":0.08734,
    "QCD_Pt_2400to3200":0.005237,
    "QCD_Pt_3200toInf" :0.000135,

    "QCD_HT50to100_TuneCP5_PSWeights"   : 187300000.0,
    "QCD_HT100to200_TuneCP5_PSWeights"   : 23640000.00,
    "QCD_HT200to300_TuneCP5_PSWeights"   : 1546000.00,
    "QCD_HT300to500_TuneCP5_PSWeights"   : 321600.00,
    "QCD_HT500to700_TuneCP5_PSWeights"   : 30980.00,
    "QCD_HT700to1000_TuneCP5_PSWeights"  : 6364.00,
    "QCD_HT1000to1500_TuneCP5_PSWeights" : 1117.00,
    "QCD_HT1500to2000_TuneCP5_PSWeights" : 108.40,
    "QCD_HT2000toInf_TuneCP5_PSWeights"  : 21.98,
        
    "QCD_bEnriched_HT100to200"  :1122000.00,
    "QCD_bEnriched_HT200to300"  :79760.00,
    "QCD_bEnriched_HT300to500"  :16600.00,
    "QCD_bEnriched_HT500to700"  :1503.000,
    "QCD_bEnriched_HT700to1000" :297.400,
    "QCD_bEnriched_HT1000to1500":48.0800,
    "QCD_bEnriched_HT1500to2000":3.95100,
    "QCD_bEnriched_HT2000toInf" :0.695700,
    
    "QCD_HT100to200_BGenFilter"  :1266000.00,
    "QCD_HT200to300_BGenFilter"  :109900.00,
    "QCD_HT300to500_BGenFilter"  :27360.00,
    "QCD_HT500to700_BGenFilter"  :2991.00,
    "QCD_HT700to1000_BGenFilter" :731.80,
    "QCD_HT1000to1500_BGenFilter":139.300,
    "QCD_HT1500to2000_BGenFilter":14.7400,
    "QCD_HT2000toInf_BGenFilter" :3.0900,

    "TTJets":831.76, # pb
}
