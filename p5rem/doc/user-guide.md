# User Guide

## Minimal Script

This is the simplest non-GUI usage pattern:

- define the remote host directly in the script
- define the remote Python command directly in the script
- create a session with a context manager
- open one remote file with a context manager
- materialise a NumPy selection locally
- disable the local cache explicitly

```python
from p5rem import bootstrap_session

REMOTE_HOST = "xfer1"
REMOTE_PYTHON = "conda run -n jas26 python"
REMOTE_FILE = "p5test/test1.nc"

with bootstrap_session(
    host=REMOTE_HOST,
    remote_python=REMOTE_PYTHON,
    login_shell=True,
    use_cache=False,
) as session:
    with session.open(REMOTE_FILE) as remote_file:
        tas = remote_file["tas"]
        first_timestep = tas[0, :, :]
        print(first_timestep.shape)
        print(first_timestep.dtype)
```

`first_timestep` is now a local NumPy array. The HDF5 metadata and chunk reads happened remotely, but the selected data has been materialised on the client side.

## Why `use_cache=False`

By default, p5rem may use a local disk cache for remote metadata and chunk payloads. If you want each run to go directly to the remote system without using any local cache, pass:

```python
use_cache=False
```

to `bootstrap_session()`.

## Running The Example

The repository includes a runnable example script:

- `examples/read_remote_slice.py`

Run it with:

```bash
/Users/bnl28/miniforge3/envs/work26/bin/python examples/read_remote_slice.py
```

Adjust the constants at the top of the script for your host, environment, file, variable, and selection.

## Notes

- If the remote Python environment is only available in shell startup files, set `login_shell=True`.
- If your environment requires setup commands first, choose one model per host.
- Module model example: `remote_setup="module load python/3.11"`, `remote_python="python"`.
- Conda/mamba activate model example: `remote_setup="source ~/miniforge3/etc/profile.d/conda.sh && conda activate jas26"`, `remote_python="python"`.
- If you use `conda run -n <env> python`, p5rem automatically adds `--no-capture-output` during bootstrap so the SSH stdio transport remains usable.
- The session and remote file both support context-manager usage and should normally be used with `with`.

### Bootstrap Troubleshooting

Bootstrap validates remote Python first, then launches the server. Distinguish
these startup failures:

- Case (a): no setup command was provided, and `remote_python` is not available.
- Case (b): no conda/mamba tooling is available, so `conda run`/`mamba run` commands fail immediately.
- Case (c): setup command exists (`remote_setup`) but fails (for example `module load ...`).

Set `P5REM_BOOTSTRAP_VERBOSE_ERRORS=1` when you need full remote stderr in the
exception text.