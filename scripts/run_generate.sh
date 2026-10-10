#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
if [ -f .env ]; then set -a; . ./.env; set +a; fi
python -m oneai.synthetic.generate "$@"
