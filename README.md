# Analysis_6b

### Preparing the environment

1. Download [miniconda3 (Linux-x86-64](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) to your favorite remote workspace. From the command line you can run: <br>
```wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh```
2. Run miniconda installer using command<br> `bash Miniconda3-latest-Linux-x86_64.sh` <br> **Note:** You may need to specify where to install. For instance, I work in EOS storage area accessed through LXPLUS so I must specify to save installation and environments in `/eos/user/s/srosenzw/`
3. Create a conda environment ([documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)) using<br> `conda create --name sixb` or use the base environment `conda activate base`
4. Some packages will require included a new source<br> `conda config --append channels conda-forge` 
5. Install the packages contained in `sixb-spec.txt` using the command<br>
```conda create --name sixb --file spec-file.txt```
6. Recommended: Add custom_scripts to your python path using<br> `conda-develop custom_scripts`

### Navigation

- Training example construction from signal event data
- Neural network training
- Neural network analysis

**Training the neural network** <br>
`nntrainer.py` must be called from the Training/ directory with the `--tag` option. Current tags consists of `parton`, `smeared`, and `reco` and possibly appended by `_6param`, `_7param_pTprod`, `_7param_dR`.<br>
