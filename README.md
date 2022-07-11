# AEPUS: 
## A tool for the Automated Extraction of Pennation angles in Ultrasound images with low Signal-to-noise ratio

# Introduction

This repository contains the work in progress on the AEPUS tool.

The current implementation performs physiological feature extraction from Ultrasound (US) images of gastrocnemius muscle.

AEPUS identifies:
- Superficial Aponeurosis (blue)
- Deep Aponeurosis (red)
- Average Fascicle inclination angle (green)

![alt text](docs/contractions.gif)

# Structure of the repository

This repository contains:

- `aepus` folder contains the source code with low-level routines.
 
- `tests` folder contains 
    - a simple test demonstrating feature extraction
    - sample US images required for the test (located in `data` directory)

# Installation and usage

To install the library we advise using [miniconda](https://docs.conda.io/en/latest/miniconda.html) package manager and creating a virtual environment.
To do it:
1. Download the repository to your local PC 
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. Move to the library directory
4. Execute the command below to create a virtual environment named `pybf_env` to install all necessary libraries listed in `conda_requirements.txt`
```bash
conda create --name aepus_env python=3.9 --file conda_requirements.txt
```
**Note:** If you have problems with installing the packages automatically you can do it manually.  Critical packages for the library are:
- numpy
- matplotlib
- scipy
- scikit-image
- pillow
- opencv (for visualization)

To use existing features we advise exploring the provided test.

To run the test: 
1. Run a terminal and activate conda environment
```bash
conda activate aepus_env
```
2. Navigate to the directory of the test
3. Execute 
```
python main.py
```

# License
All source code is released under Apache v2.0 license unless noted otherwise, please refer to the LICENSE file for details.
Example datasets under `tests/data` folder are provided under a [Creative Commons Attribution No Derivatives 4.0 International License][cc-by-nd] 

[cc-by-nd]: https://creativecommons.org/licenses/by-nd/4.0/