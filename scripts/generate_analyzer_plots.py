from trsm import *

filename = '../../NMSSM_XYH_YToHH_6b_MX_700_MY_400_6jet_testing_set_7jets_2021Aug.root'
trsm = TRSM(filename=filename)
combos = combos_6j(trsm, 7)
combos.apply_6j_model(tag)
scores = combos.scores_combo
combos.get_stats(0.8)

fig, ax = plot_combo_scores(combos)

fig, ax = plot_combo_scores_v_mass(combos)

fig, ax = plot_highest_score(combos)

fig, ax = plot_highest_score_v_mass(combos)

