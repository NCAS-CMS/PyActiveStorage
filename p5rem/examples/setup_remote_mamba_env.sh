#!/usr/bin/env bash
set -euo pipefail

# Create the minimal remote environment needed to run p5rem's remote server stub.
#
# Usage:
#   ./examples/setup_remote_mamba_env.sh [env-name]
#
# Example:
#   ./examples/setup_remote_mamba_env.sh p5rem-remote
#   conda run -n p5rem-remote python /path/to/p5rem/remote_server.py

ENV_NAME="${1:-p5rem-remote}"
MAMBA_CMD="${MAMBA_EXE:-${MAMBA_BIN:-mamba}}"

if [[ "$MAMBA_CMD" == */* ]]; then
  if [[ ! -x "$MAMBA_CMD" ]]; then
    echo "ERROR: '$MAMBA_CMD' is not executable." >&2
    echo "Set MAMBA_EXE (or MAMBA_BIN) to a valid mamba/micromamba executable." >&2
    exit 1
  fi
else
  if ! command -v "$MAMBA_CMD" >/dev/null 2>&1; then
    echo "ERROR: '$MAMBA_CMD' not found in PATH." >&2
    echo "Set MAMBA_EXE (or MAMBA_BIN) to mamba/micromamba." >&2
    exit 1
  fi
fi

echo "Creating env '$ENV_NAME' with minimal remote-server dependencies..."
"$MAMBA_CMD" create -y -n "$ENV_NAME" -c conda-forge \
  "python>=3.10" \
  "pyfive>=0.5.0" \
  "cbor2"

echo "Verifying imports in '$ENV_NAME'..."
"$MAMBA_CMD" run -n "$ENV_NAME" python - <<'PY'
import cbor2
import pyfive
import numpy
print("OK: imports successful")
print("pyfive", getattr(pyfive, "__version__", "unknown"))
print("cbor2", getattr(cbor2, "__version__", "unknown"))
print("numpy", numpy.__version__)
PY

echo
echo "Remote python command to use with p5rem:"
echo "  conda run -n $ENV_NAME python"
