

# python analyze_models.py --tag reco_6param
# python analyze_models.py --tag reco_7param_pt
# python analyze_models.py --tag reco_7param_dR
# python analyze_models.py --tag reco

# python analyze_models.py --tag=smeared_6param
# python analyze_models.py --tag=smeared_7param_pt
# python analyze_models.py --tag=smeared_7param_dR
# python analyze_models.py --tag=smeared

# python analyze_models.py --tag=parton_6param
# python analyze_models.py --tag=parton_7param_pt
# python analyze_models.py --tag=parton_7param_dR
# python analyze_models.py --tag=parton


python analyze_models.py --tag reco_6param_extranodes --nmodels="$1" --twoclass
python analyze_models.py --tag reco_7param_pt_extranodes --nmodels="$1" --twoclass
python analyze_models.py --tag reco_7param_dR_extranodes --nmodels="$1" --twoclass
python analyze_models.py --tag reco_extranodes --nmodels="$1" --twoclass

python analyze_models.py --tag reco_6param --nmodels="$1" --twoclass
python analyze_models.py --tag reco_7param_pt --nmodels="$1" --twoclass
python analyze_models.py --tag reco_7param_dR --nmodels="$1" --twoclass
python analyze_models.py --tag reco --nmodels="$1" --twoclass

python analyze_models.py --tag=parton_6param --nmodels="$1" --twoclass
python analyze_models.py --tag=parton_7param_pt --nmodels="$1" --twoclass
python analyze_models.py --tag=parton_7param_dR --nmodels="$1" --twoclass
python analyze_models.py --tag=parton --nmodels="$1" --twoclass

python analyze_models.py --tag=smeared_6param --nmodels="$1" --twoclass
python analyze_models.py --tag=smeared_7param_pt --nmodels="$1" --twoclass
python analyze_models.py --tag=smeared_7param_dR --nmodels="$1" --twoclass
python analyze_models.py --tag=smeared --nmodels="$1" --twoclass
