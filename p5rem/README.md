# p5rem

Remote HDF5/NetCDF access over SSH stdio, using a small remote pyfive-based server and a local proxy API.

## Usage

For a minimal non-GUI example that defines the remote host, setup command, and Python command directly in the script, see:

- [examples/read_remote_slice.py](examples/read_remote_slice.py)
- [doc/user-guide.md](doc/user-guide.md)

However, the basic idea is that remote files can be opened
over an ssh session, and handled locally as if they were
instances of a `pyfive.File`. There is a slightly different
approach to laziness, but otherwise `p5rem.File` quacks
just like a `pyfive.File`.

### Remote environment bootstrap (mamba)

(This is the environment on the server where you have ssh access.)
If you need to create the minimal remote Python environment for the p5rem server stub, run:

```bash
./examples/setup_remote_mamba_env.sh
```

This creates an environment (default name: `p5rem-remote`) with:

- `python>=3.10`
- `pyfive>=0.5.0`
- `cbor2`

You can choose a custom environment name:

```bash
./examples/setup_remote_mamba_env.sh my-remote-env
```

If your site uses `micromamba`, set `MAMBA_EXE` (the script also accepts
`MAMBA_BIN` for compatibility):

```bash
MAMBA_EXE=micromamba ./examples/setup_remote_mamba_env.sh my-remote-env
```

Then bootstrap a session against that interpreter, for example:

```python
from p5rem import bootstrap_session

with bootstrap_session(
    host="xfer1",
    remote_python="conda run -n my-remote-env python",
    login_shell=True,
) as session:
    ...
```

Treat `remote_setup` and `remote_python` as two separate pieces:

- `remote_setup`: optional shell fragment that prepares the environment on the remote host.
- `remote_python`: the actual Python command to run after setup has completed.

If your site requires shell setup, pick one activation model and keep it consistent for that host:

- Module-based environments: load a module, then use `python`.
- Conda/mamba environments: activate an env, then use `python`.

Example (module-based):

```python
from p5rem import bootstrap_session

with bootstrap_session(
    host="xfer1",
    remote_setup="module load jaspy",
    remote_python="python",
    login_shell=True,
) as session:
    ...
```

Example (conda/mamba activation) with a conda setup
step to make sure the conda command is available:

```python
from p5rem import bootstrap_session

with bootstrap_session(
    host="xfer1",
    remote_setup="source /path/to/conda.sh && conda activate my-remote-env",
    remote_python="python",
    login_shell=True,
) as session:
    ...
```

Example (conda run, no separate setup step):

```python
from p5rem import bootstrap_session

with bootstrap_session(
    host="xfer1",
    remote_python="conda run -n my-remote-env python",
    login_shell=True,
) as session:
    ...
```

## Testing

The test suite is split into two groups:

- Default tests: pure unit and loopback tests that do not require a real SSH target.
- Integration tests: real SSH-backed tests marked with `@pytest.mark.integration`.

The integration marker exists so the normal test run can stay fast and self-contained, while the real SSH path is still exercised explicitly when credentials and a remote test directory are available.

### Default test run

```bash
python3 -m pytest -m "not integration"
# or
make test
```

### cfdm compatibility test in a separate environment

If you want to run only the cfdm interoperability test in a dedicated env (for example `work26t`):

```bash
make test-cfdm
# or override the interpreter explicitly, eg:
make test-cfdm PYTHON_CFDM=/path/to/work26t/bin/python
```

This target uses `PYTHON_CFDM` (default: `$(PYTHON)`, which defaults to `python3`). If you need a separate environment for `cfdm`, override `PYTHON_CFDM` locally, for example via the command line or your shell environment.

### Real SSH integration run

Create a local environment override file such as `tests/testenv.sh`:

```bash
#!/usr/bin/env sh

export P5REM_SSH_HOST_ALIAS="xfer1"
export P5REM_SSH_PYTHON="conda run -n jas26 python"
export P5REM_SSH_LOGIN_SHELL="1"
export P5REM_SSH_REMOTE_DIR="p5test"
```

Then run:

```bash
source tests/testenv.sh
python3 -m pytest -m integration
# or
make test-integration
```

Notes:

- `P5REM_SSH_PYTHON` may be set to `conda run -n <env> python`; p5rem automatically adds `--no-capture-output` when bootstrapping the remote server so the SSH stdio protocol remains binary-clean.
- The remote runtime must provide `cbor2` and the required file backend: `pyfive` for HDF5/NetCDF files, `ppfive` for PP files.
- `P5REM_SSH_LOGIN_SHELL=1` wraps the remote command with `bash -lc`, which is useful on HPC systems where conda is only initialized in login-shell startup files.
- The remote round-trip test reuses the same file-comparison assertions as the loopback tests.

### Bootstrap failure cases

Bootstrap runs a remote Python preflight before launching the p5rem server. If
startup fails, the common cases are:

- Case (a): No setup command was provided, and `remote_python` is not directly available on the remote host.
- Case (b): No conda/mamba tooling is available on the remote host, so commands such as `conda run ...` cannot execute.
- Case (c): A setup command was provided (`remote_setup`), but it fails (for example `module load` fails).

Tips:

- Do not mix activation models in one command chain; pick module-load or conda/mamba-activate for a given host.
- For module-based hosts, use `remote_setup="module load ..."` with `remote_python="python"`.
- For conda/mamba activation hosts, use `remote_setup="source .../conda.sh && conda activate <env>"` (or mamba equivalent) with `remote_python="python"`.
- Use `remote_python="conda run -n <env> python"` as an alternative when you do not want a separate setup step.
- Set `P5REM_BOOTSTRAP_VERBOSE_ERRORS=1` to include remote `stderr` details in raised bootstrap errors.

### Standalone acid test

There is also a CLI wrapper for the same shared SSH round-trip assertions:

```bash
source tests/testenv.sh
python3 tests/acid_test.py
# or
make acid-test
```

This is useful for manual debugging, but the canonical automated SSH test path is the `pytest -m integration` run above.
