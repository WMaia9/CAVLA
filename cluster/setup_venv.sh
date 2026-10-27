#!/bin/bash
# Build the probe environment on the UC Merced cluster from the workstation's frozen package list
# (env/ar1_venv_freeze.txt, Python 3.12, torch 2.8 cu128). The editable Alpamayo-R1 checkout is not installed;
# its source is put on sys.path through CAVLA_R1_SRC instead.
set -e
cd "$(dirname "$0")/.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv venv --python 3.12 .venv-probe
grep -v -E '^-e ' env/ar1_venv_freeze.txt > .venv-probe/requirements.txt
uv pip install --python .venv-probe/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 -r .venv-probe/requirements.txt
