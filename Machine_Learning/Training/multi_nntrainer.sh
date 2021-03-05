# python nntrainer.py model_tag begin_index end_index config_file 
# *optional

i=$1
echo "Beginning training sessions."
while [ "$i" -le "$2" ]; do
  python nntrainer.py --tag=reco_6param_extranodes --type=reco --twoclass --outdir=twoclass --nlayers=3 --run="$i" 
  python nntrainer.py --tag=reco_7param_pt_extranodes --pTprod --type=reco --twoclass --outdir=twoclass --nlayers=3 --run="$i"
  python nntrainer.py --tag=reco_7param_dR_extranodes --DeltaR --type=reco --twoclass --outdir=twoclass --nlayers=3 --run="$i"
  python nntrainer.py --tag=reco_extranodes --pTprod --DeltaR --type=reco --twoclass --outdir=twoclass --nlayers=3 --run="$i"
  # python nntrainer.py --tag=parton_6param --run="$i" --type=parton --twoclass --outdir=twoclass
  # python nntrainer.py --tag=parton_7param_pt --pTprod --run="$i" --type=parton --twoclass --outdir=twoclass
  # python nntrainer.py --tag=parton_7param_dR --DeltaR --run="$i" --type=parton --twoclass --outdir=twoclass
  # python nntrainer.py --tag=parton --pTprod --DeltaR --run="$i" --type=parton --twoclass --outdir=twoclass
  # python nntrainer.py --tag=smeared_6param --run="$i" --type=smeared --twoclass --outdir=twoclass
  # python nntrainer.py --tag=smeared_7param_pt --pTprod --run="$i" --type=smeared --twoclass --outdir=twoclass
  # python nntrainer.py --tag=smeared_7param_dR --DeltaR --run="$i" --type=smeared --twoclass --outdir=twoclass
  # python nntrainer.py --tag=smeared --pTprod --DeltaR --run="$i" --type=smeared --twoclass --outdir=twoclass

    i=$(($i + 1))
done

echo "Beginning analysis."

python analyze_models.py --tag reco_6param_extranodes  --nmodels="$2" --twoclass --nlayers=3
python analyze_models.py --tag reco_7param_pt_extranodes  --nmodels="$2" --twoclass --nlayers=3
python analyze_models.py --tag reco_7param_dR_extranodes  --nmodels="$2" --twoclass --nlayers=3
python analyze_models.py --tag reco_extranodes  --nmodels="$2" --twoclass --nlayers=3

# python analyze_models.py --tag=parton_6param --nmodels="$2" --twoclass
# python analyze_models.py --tag=parton_7param_pt --nmodels="$2" --twoclass
# python analyze_models.py --tag=parton_7param_dR --nmodels="$2" --twoclass
# python analyze_models.py --tag=parton --nmodels="$2" --twoclass

# python analyze_models.py --tag=smeared_6param --nmodels="$2" --twoclass
# python analyze_models.py --tag=smeared_7param_pt --nmodels="$2" --twoclass
# python analyze_models.py --tag=smeared_7param_dR --nmodels="$2" --twoclass
# python analyze_models.py --tag=smeared --nmodels="$2" --twoclass
