# python nntrainer.py model_tag begin_index end_index config_file 
# *optional

i=$1
echo "Beginning training sessions."
while [ "$i" -le "$2" ]; do
  
  python nntrainer.py --run="$i" --task class --type=parton
  python nntrainer.py --run="$i" --task class --type=smeared
  python nntrainer.py --run="$i" --task class --type=reco

  i=$(($i + 1))
done

echo "Beginning analysis."

python analyze_models.py --nmodels="$2" --task class --type=reco
python analyze_models.py --nmodels="$2" --task class --type=parton
python analyze_models.py --nmodels="$2" --task class --type=smeared
