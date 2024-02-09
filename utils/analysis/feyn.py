import json

model_name = '20230731_7d266883bbfb88fe4e226783a7d1c9db_ranger_lr0.0047_batch2000_withbkg'
model_path = f'/eos/uscms/store/user/srosenzw/weaver/models/exp_sixb_official/feynnet_ranker_6b/{model_name}/predict_output'

new_model_path = "/eos/uscms/store/user/srosenzw/weaver/cmsuf/data/store/user/srosenzw/lightning/models/feynnet_lightning/X_YH_3H_6b/x3h/lightning_logs/version_23183119/predict/"
with open(f"{new_model_path}/samples.json", 'r') as file:
    mass_dict = json.load(file)




# combos = list(combinations(np.arange(6),2))
# N = len(combos)
# combo_dict = {}

# indices = []
# total = 0
# for i,combo1 in enumerate(combos):
#     combo1 = np.array(combo1)
#     for j in range(i+1, N):
#         combo2 = np.array(combos[j])
#         # print(combo,combos[j])
#         if combo1[0] in combo2 or combo1[1] in combo2: continue
#         for k in range(j+1, N):
#             combo3 = np.array(combos[k])
#             if combo2[0] in combo3 or combo2[1] in combo3: continue
#             if combo1[0] in combo3 or combo1[1] in combo3: continue
#             indices.append(np.concatenate((combo1, combo2, combo3)))
#             combo_dict[total] = np.concatenate((combo1, combo2, combo3))
#             # print(f"combo_dict[{total}] = {combo_dict[total]}")
#             # combo_dict[total] = 
#             total += 1
#         # break
#     # break
# indices = np.row_stack([ind for ind in indices])