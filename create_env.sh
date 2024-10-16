module --force purge
module load StdEnv/2020  gcc/9.3.0 opencv/4.8.0
module load python/3.10.2

echo "loading module done"

cwd=$(pwd)

echo "creating new virtualenv"

virtualenv ~/$1 
source ~/$1/bin/activate

cd ../../

pip install --no-index --upgrade pip

pip install --no-index numpy
pip install mmcv==1.7.1
pip install tqdm
pip install sklearn
pip install jupyterlab
pip install ipywidgets
pip install icecream
pip install --no-index wandb==0.16.0
pip install matplotlib
pip install xarray
pip install h5netcdf
pip install torch torchvision torchmetrics torch-summary
pip install seaborn