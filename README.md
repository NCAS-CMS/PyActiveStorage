[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Documentation Status](https://app.readthedocs.org/projects/pyactivestorage/badge/?version=latest)](https://pyactivestorage.readthedocs.io/en/latest/?badge=latest)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Test](https://github.com/NCAS-CMS/PyActiveStorage/actions/workflows/run-tests.yml/badge.svg)](https://github.com/NCAS-CMS/PyActiveStorage/actions/workflows/run-tests.yml)
[![codecov](https://codecov.io/gh/NCAS-CMS/PyActiveStorage/graph/badge.svg?token=1olGjnvAOp)](https://codecov.io/gh/NCAS-CMS/PyActiveStorage)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pyactivestorage/badges/version.svg)](https://anaconda.org/conda-forge/pyactivestorage)

![pyactivestoragelogo](https://raw.githubusercontent.com/NCAS-CMS/PyActiveStorage/main/doc/figures/PyActiveStorage-logo-complete.jpg)

## PyActiveStorage

- [Latest documentation on ReadTheDocs](https://pyactivestorage.readthedocs.io/en/latest/)
- [Project CI (GitHub Actions)](https://github.com/NCAS-CMS/PyActiveStorage/actions)
- [conda-forge feedstock](https://github.com/conda-forge/pyactivestorage-feedstock)
- Testing and CI details: [README_testing.md](README_testing.md)

PyActiveStorage lets you run reductions (``min``, ``max``, ``sum``, ``mean``)
against chunked NetCDF/HDF5 data without always pulling full arrays over the
network.

The package is primarily used in two deployment models:

1. **Object storage + adjacent Reductionist**
   - Your data lives on S3-compatible object storage (or HTTPS object-serving).
   - A Reductionist service is deployed near that storage.
   - Your client sends reduction requests to Reductionist.
2. **SSH-accessible remote storage (p5rem)**
   - You can SSH to a remote machine where the files exist.
   - PyActiveStorage bootstraps a lightweight remote server over SSH stdio.
   - Reductions run using remote file access via ``rFile`` / ``rDataset``.

## Install

### Conda environment

```bash
conda install -c conda-forge mamba
mamba env create -n activestorage -f environment.yml
conda activate activestorage
```

### Install package

```bash
pip install -e .
```

Python versions supported: 3.10, 3.11, 3.12, 3.13.

## User-facing API

The main API entry point is ``Active``:

```python
from activestorage.active import Active

active = Active(dataset_or_uri, ncvar="tas")
result = active.mean(axis=(0, 1))[:]
```

## Workflow A: S3/HTTPS + Reductionist

Use this when you have object storage that is paired with a Reductionist
endpoint.

### A1. S3 example

```python
from activestorage.active import Active

test_file_uri = "my-bucket/path/to/file.nc"

active = Active(
	test_file_uri,
	ncvar="tas",
	interface_type="s3",
	storage_options={
		"key": "<s3-access-key>",
		"secret": "<s3-secret-key>",
		"client_kwargs": {
			"endpoint_url": "https://my-s3-endpoint.example"
		},
	},
	active_storage_url="https://my-reductionist.example/",
	option_disable_chunk_cache=True,
)

result = active.min(axis=(0, 1))[:]
print(result.shape)
```

### A2. HTTPS example

```python
from activestorage.active import Active

uri = "https://data.example.org/path/to/file.nc"

active = Active(
	uri,
	ncvar="ta",
	interface_type="https",
	active_storage_url="https://my-reductionist.example/",
)

result = active.mean(axis=(0, 1))[:]
print(result)
```

## Workflow B: SSH remote storage via p5rem

Use this when you can SSH to the host that has the file, and that remote Python
environment has ``cbor2`` plus the file backend (``pyfive`` for HDF5/NetCDF,
or ``ppfive`` for PP).

```python
from activestorage.active import Active
from activestorage.bootstrap import bootstrap_session

with bootstrap_session(
	host="hpc-login",
	remote_setup="module load jaspy",
	remote_python="python",
	login_shell=True,
) as session:
	with session.open("/remote/path/data.nc") as f:
		ds = f["tas"]
		active = Active(ds, interface_type="p5rem")
		result = active.max(axis=(0, 1))[:]
		print(result)
```

## Choosing storage type

Storage type can be inferred from URI in many cases, but for remote/object
workflows it is clearer to set it explicitly via ``interface_type``:

- ``"s3"`` for S3 + Reductionist
- ``"https"`` for HTTPS + Reductionist
- ``"p5rem"`` for SSH-backed remote proxy datasets

## Documentation

Project documentation is hosted on ReadTheDocs:

- https://pyactivestorage.readthedocs.io/en/latest/

To build locally:

```bash
sphinx-build -Ea -b html doc doc/build/html
```

Then open ``doc/build/html/index.html``.

## Testing, CI, and coverage

Testing commands, CI notes, and performance-testing details are now in
[README_testing.md](README_testing.md).
