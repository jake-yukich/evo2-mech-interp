```
conda env create -f environment.yaml # pip will fail
conda activate evo2

conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge transformer-engine-torch=2.3.0
pip install flash-attn==2.8.0.post2 --no-build-isolation

pip install evo2
```

pip whack-a-mole:
```
ipykernel
plotly
scipy
jaxtyping
ete3
datasets
biopython
umap
matplotlib
```

setup root `data/gtdb/` directory, add the metadata and tree files from GTDB
- TODO: setup a script to do this with progress bar, etc.
- (files can be found [here](https://data.gtdb.aau.ecogenomic.org/releases/release220/220.0/))

