# python nntrainer.py <config> <run_code> <hyperparameter_name> <hyperparameter_value>

i=1
while [ "$i" -le "$1" ]; do
  python nntrainer.py ../Config/nn_hyperparameters2.cfg "$i" hidden_layers 3
  i=$(($i + 1))
done
