mamba create -n neuralgcm python=3.10.16
salloc -p gpu 
mamba install -y cuda cudatoolkit cudnn -c nvidia --force-reinstall
mamba config --add channels conda-forge 
mamba install -y cudnn -c nvidia conda-forge #if cudann is not correct
pip install -U jax
pip install -U "jax[cuda12_local]"
pip install "xarray[complete]"  
pip install -q -U neuralgcm dinosaur-dycore gcsfs

mamba install jupyter ipykernel
python -m ipykernel install --user --name neuralgcm2 --display-name neuralgcm2
