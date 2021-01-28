# python nntrainer.py <config> <run_code> <hyperparameter_name> <hyperparameter_value>

i=1
while [ "$i" -le "$1" ]; do
  python nntrainer.py ../Config/nn_hyperparameters.cfg "$i" hidden_layers 9
  i=$(($i + 1))
done