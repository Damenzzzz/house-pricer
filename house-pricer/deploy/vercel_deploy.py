from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "house-pricer"


def backend_url() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1].rstrip("/")
    p = ROOT / "var" / "tunnel_url.txt"
    if p.exists() and p.read_text().strip():
        return p.read_text().strip().rstrip("/")
    sys.exit("укажите адрес бэкенда: python deploy/vercel_deploy.py https://...")


def _env_value(key: str) -> str | None:
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def main():
    backend = backend_url()
    out = Path(tempfile.mkdtemp(prefix="house-pricer-vercel-"))
    shutil.copytree(ROOT / "web", out, dirs_exist_ok=True)
    (out / "vercel.json").write_text(json.dumps({
        "cleanUrls": True,
        "trailingSlash": False,
        "rewrites": [
            {"source": "/api/:path*", "destination": f"{backend}/api/:path*"},
            {"source": "/healthz", "destination": f"{backend}/healthz"},
        ],
        "headers": [
            {"source": "/static/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=3600"}]},
            {"source": "/(.*)", "headers": [
                {"key": "X-Content-Type-Options", "value": "nosniff"},
                {"key": "Referrer-Policy", "value": "strict-origin-when-cross-origin"}]},
        ],
    }, ensure_ascii=False, indent=2))
    cmd = ["npx", "--yes", "vercel@latest", "deploy", "--prod", "--yes", "--name", PROJECT]
    token = _env_value("VERCEL_TOKEN")
    if token:
        cmd += ["--token", token]
    print(f"бэкенд: {backend}\nсборка: {out}")
    r = subprocess.run(cmd, cwd=out, capture_output=True, text=True)
    urls = [w for w in (r.stdout + r.stderr).split() if w.startswith("https://") and "vercel.app" in w]
    if r.returncode != 0:
        sys.exit((r.stderr or r.stdout)[-2000:])
    print("готово:", urls[-1] if urls else r.stdout.strip()[-300:])


if __name__ == "__main__":
    main()
