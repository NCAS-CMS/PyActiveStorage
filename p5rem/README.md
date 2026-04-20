# p5rem

Remote HDF5/NetCDF access over SSH stdio, using a small remote pyfive-based server and a local proxy API.

## Usage

For a minimal non-GUI example that defines the remote host and Python command directly in the script, see:

- [examples/read_remote_slice.py](examples/read_remote_slice.py)
- [doc/user-guide.md](doc/user-guide.md)

### Remote environment bootstrap (mamba)

If you need to create the minimal remote Python environment for the p5rem
server stub, run:

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

Then point p5rem at that interpreter, for example:

```bash
export P5REM_SSH_PYTHON="conda run -n my-remote-env python"
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
- `P5REM_SSH_LOGIN_SHELL=1` wraps the remote command with `bash -lc`, which is useful on HPC systems where conda is only initialized in login-shell startup files.
- The remote round-trip test reuses the same file-comparison assertions as the loopback tests.

### Standalone acid test

There is also a CLI wrapper for the same shared SSH round-trip assertions:

```bash
source tests/testenv.sh
python3 tests/acid_test.py
# or
make acid-test
```

This is useful for manual debugging, but the canonical automated SSH test path is the `pytest -m integration` run above.
