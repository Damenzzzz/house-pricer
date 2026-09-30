import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_tmp = tempfile.mkdtemp(prefix="baga-test-")
os.environ["BAGA_VAR"] = _tmp
os.environ["BAGA_USERS"] = str(Path(_tmp) / "users.json")
os.environ["LANGFUSE_TRACING"] = "false"
os.environ["TRACING"] = "0"
os.environ["COOKIE_SECURE"] = "0"
