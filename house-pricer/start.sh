#!/bin/sh
set -e
python qdrant_store.py ensure
if [ "${RUN_BOT:-0}" = "1" ] && [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
  python telegram_bot.py &
fi
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8501}" --proxy-headers --forwarded-allow-ips="*"
