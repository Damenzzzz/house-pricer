import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("KRISHA_DATA", ROOT / "data"))
LISTINGS_RAW = Path(os.getenv("LISTINGS_RAW", DATA_DIR / "listings_raw.parquet"))
PHOTOS_DIR = Path(os.getenv("PHOTOS_DIR", DATA_DIR / "photos"))

ART_DIR = Path(os.getenv("KRISHA_ARTIFACTS", ROOT / "artifacts"))
LISTINGS_PATH = ART_DIR / "listings.parquet"
PHOTOS_INDEX = ART_DIR / "photos.parquet"
SHARDS_DIR = ART_DIR / "emb_shards"
EMB_PATH = ART_DIR / "photo_emb.npy"
VALID_PATH = ART_DIR / "photo_valid.npy"
EMB_META = ART_DIR / "photo_emb.json"
ROOMS_PATH = ART_DIR / "photo_rooms.parquet"
DESC_EMB_PATH = ART_DIR / "desc_emb.npy"
DESC_IDS_PATH = ART_DIR / "desc_ids.npy"

CLIP_MODEL_ID = os.getenv("CLIP_MODEL_ID", "google/siglip2-base-patch16-224")

VIEWS = tuple(os.getenv("VIEWS", "pad").split(","))

USE_CLAHE = os.getenv("USE_CLAHE", "0") == "1"

SHARD_SIZE = 4096
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", min(8, os.cpu_count() or 1)))

RENDER_THR = 0.7
ROOM_AUTO_CONF = 0.5

TEXT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-1024")
TEXT_EMBED_URL = os.getenv("ALEM_URL")
TEXT_EMBED_KEY = os.getenv("EMBED_API_KEY")

CHAT_MODEL = os.getenv("ALEM_CHAT_MODEL", "alemllm")
CHAT_URL = os.getenv("ALEM_URL")
CHAT_KEY = os.getenv("ALEM_API_KEY")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.0))
CHAT_TOP_P = float(os.getenv("CHAT_TOP_P", 1.0))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", 800))

LLM_CHAIN = [p.strip() for p in os.getenv("LLM_CHAIN", "gemini,alem").split(",") if p.strip()]
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_THINKING = os.getenv("GEMINI_THINKING", "minimal")
JUDGE_CHAIN = [p.strip() for p in os.getenv("JUDGE_CHAIN", "alem,gemini").split(",") if p.strip()]
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-lite-preview-tts")
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", 30))
LLM_DAILY_BUDGET_USD = float(os.getenv("LLM_DAILY_BUDGET_USD", 3.0))
LLM_CACHE = os.getenv("LLM_CACHE", "1") == "1"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", 0.94))

VAR_DIR = Path(os.getenv("HOUSE_PRICER_VAR", ROOT / "var"))
VAR_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = VAR_DIR / "house_pricer.sqlite"
USERS_PATH = Path(os.getenv("HOUSE_PRICER_USERS", VAR_DIR / "users.json"))

RRF_K = 60
RRF_DESC_WEIGHT = float(os.getenv("RRF_DESC_WEIGHT", 0.5))
SEED = 0


try:
    import torch
except ImportError:
    torch = None


def _device():
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _device()
DTYPE = None if torch is None else (torch.float16 if DEVICE == "cuda" else torch.float32)
