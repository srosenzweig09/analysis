# Analysis_6b

I run from LXPLUS using a CONDA environment saved in a YML file (sixb.yml).
RECOMMENDATION: Clone conda environment for  best results

1. [Download miniconda3](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) (Linux 64-bit with Python 3.8)
2. Create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) using `conda create --name sixb`
3. Install the following packages either by including them after the sixb in the above command or use `conda install numpy uproot3 uproot3-methods awkward0 sklearn matplotlib keras tensorflow pandas conda-develop`
4. Add Analysis_6b/custom_scripts to your python path using `conda-develop Analysis_6b/custom_scripts`
