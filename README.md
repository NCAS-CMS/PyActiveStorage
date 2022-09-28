## Active Storage Prototype

### Create virtual environment

```bash
(base) conda install -c conda-forge mamba
(base) mamba env create -n activestorage -f environment.yml
conda activate zarr-kerchunk
```

### Install with `pip`

```bash
pip install -e .
```

### Run tests

```bash
pytest -n 2
```
