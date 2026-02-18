#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
python -m pip install --quiet "pip<25"

# make sure pip-tools itself is recent enough
python -m pip install --quiet -U "pip-tools>=7.4"

# ------------------------------------------------------------------
# 1) save only PyPI packages (no file:// or git+ references)
pip freeze --exclude-editable | grep '==' > requirements.in

# 2) compile with hashes
pip-compile --generate-hashes \
            --output-file requirements.txt \
            requirements.in

echo "✓  requirements.txt written with $(grep -c '==' requirements.txt) packages"
