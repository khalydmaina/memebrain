"""Configuration: env loading and constants."""

import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Keys (Telegram + Groq only; all market data sources are keyless)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")
GROQ_MODEL_SMART = os.getenv("GROQ_MODEL_SMART", "llama-3.3-70b-versatile")

# Paper portfolio
PAPER_START_SOL = float(os.getenv("PAPER_START_SOL", "10"))
POSITION_SIZE_SOL = float(os.getenv("POSITION_SIZE_SOL", "0.25"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "6"))
MAX_NEW_POSITIONS_PER_DAY = int(os.getenv("MAX_NEW_POSITIONS_PER_DAY", "12"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "-30"))
TP1_PCT = float(os.getenv("TP1_PCT", "60"))       # sell half
TP2_PCT = float(os.getenv("TP2_PCT", "150"))      # sell the rest
MAX_HOLD_HOURS = float(os.getenv("MAX_HOLD_HOURS", "24"))

# Entry gates (reflex layer, non-negotiable)
MIN_LIQUIDITY_USD = float(os.getenv("MIN_LIQUIDITY_USD", "10000"))
MIN_VOLUME_H1_USD = float(os.getenv("MIN_VOLUME_H1_USD", "2000"))
MAX_TOP10_HOLDER_PCT = float(os.getenv("MAX_TOP10_HOLDER_PCT", "45"))
MIN_BUY_SCORE = float(os.getenv("MIN_BUY_SCORE", "62"))
MAX_TOKEN_AGE_HOURS = 48.0     # beyond this it's not our hunting ground
FRESH_AGE_HOURS = 6.0          # <6h = "fresh", 6-48h = "young"

# Scan cadence
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "180"))
GECKO_PAGES = 2                # new_pools pages per scan (20 pools each)

# Outcome checkpoints (minutes after decision) used as learning labels
OUTCOME_CHECKPOINTS_MIN = (5, 30, 120, 1440)
# A decision counts as a WIN if price peaks >= +30% at any checkpoint
# without rugging; a LOSS if it rugs or drops 30% first.
WIN_THRESHOLD = 1.30
LOSS_THRESHOLD = 0.70
RUG_THRESHOLD = 0.15           # price at <=15% of entry == rugged/dead

DB_FILE = os.path.join(BASE_DIR, "memebrain.db")
LOG_FILE = os.path.join(BASE_DIR, "memebrain.log")

NARRATIVE_BUCKETS = [
    "dog", "cat", "frog", "ai", "political", "celebrity", "food",
    "brainrot", "degen", "event", "parody", "other",
]


def validate_config(need_telegram=True):
    missing = []
    if need_telegram and not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if need_telegram and not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")
    if missing:
        raise SystemExit(f"[CONFIG] Missing in .env: {', '.join(missing)}")
