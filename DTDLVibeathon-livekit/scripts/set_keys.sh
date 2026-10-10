#!/usr/bin/env bash
# Writes LiveKit (and optional OpenAI) keys to .env. Usage: scripts/set_keys.sh
set -euo pipefail
cd "$(dirname "$0")/.."

trim() { printf '%s' "$1" | tr -d " \t\r\n\"'"; }

read -r -p "LiveKit URL (wss://...): " url
read -r -p "LiveKit API key: " key
read -r -s -p "LiveKit API secret (hidden): " secret; echo
read -r -s -p "OpenAI API key (hidden, Enter to skip): " openai; echo

url=$(trim "$url"); key=$(trim "$key"); secret=$(trim "$secret"); openai=$(trim "$openai")

umask 077
cat > .env <<EOF
LIVEKIT_URL=$url
LIVEKIT_API_KEY=$key
LIVEKIT_API_SECRET=$secret
OPENAI_API_KEY=$openai
HF_HUB_OFFLINE=1
PYTHONWARNINGS=ignore
EOF

echo "Saved .env: URL ${#url} chars, key ${#key} chars (starts ${key:0:3}), secret ${#secret} chars, OpenAI key ${#openai} chars."
