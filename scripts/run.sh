#!/usr/bin/env bash
# Usage: scripts/run.sh [generate|pipeline|api|mock|ab|eval|livekit] [extra args]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
cmd="${1:-pipeline}"
shift || true
case "$cmd" in
  generate) python -m oneai.synthetic.generate "$@" ;;
  pipeline) python -m oneai.offline.pipeline "$@" ;;
  api) .venv/bin/uvicorn oneai.realtime.app:app --port 8000 "$@" ;;
  mock) python -m oneai.realtime.mock_replay "$@" ;;
  ab) python -m oneai.ab_demo "$@" ;;
  eval) python -m oneai.eval "$@" ;;
  livekit) python -m oneai.livekit_adapter.agent dev "$@" ;;
  *) echo "unknown command: $cmd" >&2; exit 1 ;;
esac
