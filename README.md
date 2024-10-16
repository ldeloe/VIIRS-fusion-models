# VIIRS-fusion-models

## Dependencies
The following dependencies were used to create the current venv:
- python/3.10.2
- StdEnv/2020  gcc/9.3.0 opencv/4.8.0
- pip install --no-index numpy
- pip install mmcv==1.7.1
- pip install tqdm
- pip install sklearn
- pip install jupyterlab
- pip install ipywidgets
- pip install icecream
- pip install --no-index wandb==0.16.0
- pip install matplotlib
- pip install xarray
- pip install h5netcdf
- pip install torch torchvision torchmetrics torch-summary
- pip install seaborn

## Cloning the repo:

Clone this repo by using `git clone <link_to_repo>`

## Create a new environment

Compute Canada/Digital Research Alliance does NOT have support for Conda environments. Instead, we use the inbuilt [venv](https://docs.python.org/3/library/venv.html) module to create new environments.

The repo contains a [create_env.sh](create_env.sh) which will create a virtual environment for you in **compute canada**.

To create a new environment `bash create_env.sh <envname>`.
<br/> This will create a new env in the `~/<envname>` folder, which is nothing but root folder.

## Activating the environment

To activate the env, use the command `source ~/<envname>/bin/activate`. 

## Running Jobs on Compute Canada

Submit jobs from `/home/lcbdeloe/projects/def-ka3scott/VIIRS-fusion-models/compute_canada/submit`