# Analysis_6b

### Preparing the environment

1. [Download miniconda3](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) (Linux 64-bit with Python 3.8) to your favorite remote workspace
2. Run miniconda installer using <br> `bash Miniconda3-latest-Linux-x86_64.sh`
3. Create a conda environment ([documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)) using<br> `conda create --name sixb`
4. Some packages will require included a new source<br> `conda config --append channels conda-forge`
5. Install the following packages either by including them after the sixb in the above command or use<br> `conda install numpy uproot3 uproot3-methods awkward0 sklearn matplotlib keras tensorflow pandas conda-develop`
6. Add Analysis_6b/custom_scripts to your python path using<br> `conda-develop Analysis_6b/custom_scripts`
