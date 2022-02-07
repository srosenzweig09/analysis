## Table of Contents
- [Table of Contents](#table-of-contents)
- [1. Preparing conda environment](#1-preparing-conda-environment)
- [2. From Gen Productions to Sensitivities](#2-from-gen-productions-to-sensitivities)
  - [1. Generate gridpacks](#1-generate-gridpacks)
  - [2. Hadronize and Simulate Detector Response](#2-hadronize-and-simulate-detector-response)
  - [3. Run skims](#3-run-skims)
  - [4. Generate Background Estimation Regions](#4-generate-background-estimation-regions)
  - [5. Run Combine](#5-run-combine)
<!-- 4. [Analyzing the network performance](#4.-analyzing-network-performance) -->



## 1. Preparing conda environment

1. Download [miniconda3 (Linux-x86-64](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) to your favorite remote workspace. From the command line you can run: <br>
 ```wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh```
2. Run miniconda installer using command<br> `bash Miniconda3-latest-Linux-x86_64.sh` <br> **Note:** You may need to specify where to install. For instance, I work in EOS storage area accessed through LXPLUS so I must specify to save installation and environments in `/eos/user/s/srosenzw/`
3. Create a conda environment ([documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)) using<br> `conda create --name sixb` or use the base environment `conda activate base`
4. Some packages will require included a new source<br> `conda config --append channels conda-forge` 
5. Install the packages contained in `sixb-spec.txt` using the command<br>
 ```conda create --name sixb --file spec-file.txt```
 6. Recommended: Clone and add modules/ to your python path using<br> `conda develop modules`


## 2. From Gen Productions to Sensitivities

### 1. Generate gridpacks
- See [sixB repo on GitHub](https://github.com/srosenzweig09-forks/sixB)
 - See [MadGraph/gridpacks](https://github.com/srosenzweig09-forks/sixB/tree/master/MadGraph/gridpacks)
  - See [genproductions](https://github.com/cms-sw/genproductions)
  - See [tutorial on producing gridpacks](https://twiki.cern.ch/twiki/bin/viewauth/CMS/QuickGuideMadGraph5aMCatNLO#Quick_tutorial_on_how_to_produce)

### 2. Hadronize and Simulate Detector Response
 - See [sixB/FullSim](https://github.com/srosenzweig09-forks/sixB/tree/master/FullSim)

### 3. Run skims
- See [sixB/analysis/sixBanalysis](https://github.com/srosenzweig09-forks/sixB/tree/master/analysis/sixBanalysis)

### 4. Generate Background Estimation Regions

- Generate a config file (e.g. [config/regionConfig.cfg](config/regionConfig.cfg)) then run the following command: 

```
python scripts/skimRegions.py --cfg /path/to/cfg
```

### 5. Run Combine

See [HiggsAnalysis-CombinedLimit](https://github.com/srosenzweig09-forks/HiggsAnalysis-CombinedLimit)