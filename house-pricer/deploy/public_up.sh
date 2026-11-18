#!/bin/sh
set -e
cd "$(dirname "$0")/.."
docker compose --profile telegram up -d
until curl -fs http://localhost:8501/healthz >/dev/null; do sleep 2; done
pkill -f "cloudflared tunnel" 2>/dev/null || true
mkdir -p var/logs
nohup caffeinate -is "$HOME/.local/bin/cloudflared" tunnel --no-autoupdate --url http://localhost:8501 \
  > var/logs/tunnel.log 2>&1 &
URL=""
while [ -z "$URL" ]; do sleep 2; URL=$(grep -aoE "https://[a-z0-9-]+\.trycloudflare\.com" var/logs/tunnel.log | head -1); done
echo "$URL" > var/tunnel_url.txt
echo "туннель: $URL"
until curl -fs "$URL/healthz" >/dev/null; do sleep 3; done
python3 deploy/vercel_deploy.py "$URL"
