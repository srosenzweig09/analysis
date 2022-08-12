#!/bin/sh

INS=$1
ENV=$2

echo "Using installer:  $(which ${INS})"
echo "Creating new env: ${ENV}"

read -p "Do you want to continue? [Y/y]" -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi

$INS create -n $ENV python==3.9 

$INS activate $ENV

$INS install pytorch==1.12.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
pip install torch-geometric
pip install apex==0.9.10dev awkward==1.8.0  awkward0==0.15.5 matplotlib==3.5.3 numpy==1.23.1 onnxruntime==1.12.1 PyYAML==6.0 scikit_learn==1.1.2 scipy==1.9.0 tables==3.7.0 tqdm==4.64.0 uproot==4.3.4 uproot3==3.14.4 vector==0.8.5