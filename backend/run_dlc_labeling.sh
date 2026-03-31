#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_PYTHON="${DLC_PYTHON:-/Users/noemiamahmud/miniconda3/envs/rat-surveillance/bin/python}"

if [[ ! -x "$CONDA_PYTHON" ]]; then
  echo "DLC Python not found at: $CONDA_PYTHON" >&2
  echo "Set DLC_PYTHON to the correct interpreter path and retry." >&2
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

cd "$SCRIPT_DIR"
exec "$CONDA_PYTHON" -m app.dlc_labeling "$@"
