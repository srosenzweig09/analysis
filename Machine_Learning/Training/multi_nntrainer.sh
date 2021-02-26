# python nntrainer.py model_tag begin_index end_index config_file 
# *optional
echo "$1"
echo "$2"

i=$1
while [ "$i" -le "$2" ]; do
#   python nntrainer.py --tag=reco_6param --run="$i" --type=reco
#   python nntrainer.py --tag=reco_7param_pt --pTprod --run="$i" --type=reco
#   python nntrainer.py --tag=reco_7param_dR --DeltaR --run="$i" --type=reco
#   python nntrainer.py --tag=reco --pTprod --DeltaR --run="$i" --type=reco
#   python nntrainer.py --tag=parton_6param --run="$i" --type=parton
#   python nntrainer.py --tag=parton_7param_pt --pTprod --run="$i" --type=parton
#   python nntrainer.py --tag=parton_7param_dR --DeltaR --run="$i" --type=parton
#   python nntrainer.py --tag=parton --pTprod --DeltaR --run="$i" --type=parton
  python nntrainer.py --tag=smeared_6param --run="$i" --type=smeared
  python nntrainer.py --tag=smeared_7param_pt --pTprod --run="$i" --type=smeared
  python nntrainer.py --tag=smeared_7param_dR --DeltaR --run="$i" --type=smeared
  python nntrainer.py --tag=smeared --pTprod --DeltaR --run="$i" --type=smeared

    i=$(($i + 1))
done



# python analyze_models.py --tag reco_6param --nmodels="$2"
# python analyze_models.py --tag reco_7param_pt --nmodels="$2"
# python analyze_models.py --tag reco_7param_dR --nmodels="$2"
# python analyze_models.py --tag reco --nmodels="$2"

# python analyze_models.py --tag=parton_6param --nmodels="$2"
# python analyze_models.py --tag=parton_7param_pt --nmodels="$2"
# python analyze_models.py --tag=parton_7param_dR --nmodels="$2"
# python analyze_models.py --tag=parton --nmodels="$2"

python analyze_models.py --tag=smeared_6param --nmodels="$2"
python analyze_models.py --tag=smeared_7param_pt --nmodels="$2"
python analyze_models.py --tag=smeared_7param_dR --nmodels="$2"
python analyze_models.py --tag=smeared --nmodels="$2"