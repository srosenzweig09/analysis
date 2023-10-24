import subprocess
import numpy as np
from itertools import combinations

# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/XY_3H_reco_ranker/20230220_ranger_lr0.0047_batch1024__withbkg/predict_output'
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/XY_3H_reco_ranker/20230303_ranger_lr0.0047_batch1024__maxbtag_sigma_25_withbkg/predict_output'
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/XY_3H_reco_ranker/20230305_ranger_lr0.0047_batch1024__maxbtag_sigma_25_withbkg/predict_output'
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/XY_3H_reco_ranker/20230306_ranger_lr0.0047_batch1024__maxbtag_sigma_25_bkg_loss_1_withbkg/predict_output'
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/XY_3H_reco_ranker/20230307_ranger_lr0.0047_batch1024__maxbtag_sigma_25_bkg_loss_2_withbkg/predict_output'
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/XY_3H_reco_ranker/20230315_ranger_lr0.0047_batch1024__MX_450_550_700_900_1000_sigma_25_bkg_loss_1_withbkg/predict_output'
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230407_ranger_lr0.0047_batch1024_/predict_output'
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230413_ranger_lr0.0047_batch1024__updated_signal_rank_withbkg/predict_output'
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230413_ranger_lr0.0047_batch1024__updated_signal_rank_withbkg/predict_output'
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230413_ranger_lr0.0047_batch1024__5_mass_pts_100_epochs_withbkg/predict_output'
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230414_ranger_lr0.0047_batch1024__13_mass_points_100_epochs_withbkg/predict_output'
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230504_ranger_lr0.0047_batch1024__lightning_100epochs_NO_reweighting_withbkg/predict_output' # NO reweighting
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230504_ranger_lr0.0047_batch1024__lightning_100epochs_reweighting_withbkg/predict_output' # w/ reweighting
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230524_ranger_lr0.0047_batch1024__39_masses_with_reweighting_withbkg/predict_output' # w/ reweighting, 39 masses
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230524_ranger_lr0.0047_batch1024__39_masses_no_reweighting_withbkg/predict_output' # w/ reweighting, 39 masses
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230614_ranger_lr0.0047_batch1024__74_masses_with_mxmy_reweighting_withbkg/predict_output' # w/ reweighting, 74 masses
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230614_ranger_lr0.0047_batch1024__74_masses_with_mx_reweighting_withbkg/predict_output' # w/ reweighting, 74 masses
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230608_ranger_lr0.0047_batch1024__74_masses_no_mxmy_reweighting_withbkg_withbkg/predict_output' # wout/ reweighting, 74 masses
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230619_ranger_lr0.0047_batch1024__withbkg/predict_output' # 
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230621_ranger_lr0.0047_batch1024__99_masses_mxmy_reweighting_withbkg/predict_output' # 
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230621_ranger_lr0.0047_batch1024__99_masses_no_reweighting_withbkg/predict_output' # 
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230626_ranger_lr0.0047_batch1024__100_masses_mxmy_reweighting_withbkg/predict_output'  
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230626_ranger_lr0.0047_batch1024__100_masses_no_reweighting_withbkg/predict_output'  
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230626_ranger_lr0.0047_batch1024__100_masses_mx_reweighting_withbkg/predict_output'  
# model_path = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/sixb/weaver-multiH/weaver/models/exp_xy/feynnet_6b/20230628_ranger_lr0.0047_batch5000__no_signal_withbkg/predict_output'  


model_path = '/eos/uscms/store/user/srosenzw/weaver/models/exp_sixb_official/feynnet_ranker_6b/20230731_7d266883bbfb88fe4e226783a7d1c9db_ranger_lr0.0047_batch2000_withbkg/predict_output/'

output = subprocess.check_output(['ls',model_path])
output = output.decode('UTF-8').split('\n')
output = [out for out in output if out != '']

combos = list(combinations(np.arange(6),2))
N = len(combos)
combo_dict = {}

indices = []
total = 0
for i,combo1 in enumerate(combos):
    combo1 = np.array(combo1)
    for j in range(i+1, N):
        combo2 = np.array(combos[j])
        # print(combo,combos[j])
        if combo1[0] in combo2 or combo1[1] in combo2: continue
        for k in range(j+1, N):
            combo3 = np.array(combos[k])
            if combo2[0] in combo3 or combo2[1] in combo3: continue
            if combo1[0] in combo3 or combo1[1] in combo3: continue
            indices.append(np.concatenate((combo1, combo2, combo3)))
            combo_dict[total] = np.concatenate((combo1, combo2, combo3))
            # print(f"combo_dict[{total}] = {combo_dict[total]}")
            # combo_dict[total] = 
            total += 1
        # break
    # break
indices = np.row_stack([ind for ind in indices])