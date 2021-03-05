# Apply preselection
echo "Running script to generate training examples for partons WITH preselections"
python collect_training_examples.py --type=parton  --presel
echo "Running script to generate training examples for reco WITH preselections"
python collect_training_examples.py --type=reco --presel

# Don't apply preselection
echo "Running script to generate training examples for partons WITHOUT preselections"
python collect_training_examples.py --type=parton
echo "Running script to generate training examples for reco WITHOUT preselections"
python collect_training_examples.py --type=reco