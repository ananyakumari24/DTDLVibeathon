#!/usr/bin/env bash
# Usage: scripts/run.sh [generate|pipeline|api|mock|ab|eval|livekit] [extra args]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
if [ -f .env ]; then set -a; . ./.env; set +a; fi
# Use the project's venv even when it hasn't been activated.
if [ -x .venv/bin/python ]; then
  py=.venv/bin/python
else
  py=python3
fi
cmd="${1:-pipeline}"
shift || true
case "$cmd" in
  generate) "$py" -m oneai.synthetic.generate "$@" ;;
  pipeline) "$py" -m oneai.offline.pipeline "$@" ;;
  api) "$py" -m uvicorn oneai.realtime.app:app --port 8000 "$@" ;;
  mock) "$py" -m oneai.realtime.mock_replay "$@" ;;
  ab) "$py" -m oneai.ab_demo "$@" ;;
  eval) "$py" -m oneai.eval "$@" ;;
  livekit) "$py" -m oneai.livekit_adapter.agent dev "$@" ;;
  *) echo "unknown command: $cmd" >&2; exit 1 ;;
esac
