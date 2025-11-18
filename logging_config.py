import os
import logging
from pathlib import Path
import valkey  # Redis/Valkey client
from logtail import LogtailHandler
from dotenv import load_dotenv

# Load environment variables from .env file if not in a rendering environment
if os.getenv("RENDER") != "true":
    load_dotenv()

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

LOGTAIL_TOKEN = os.environ.get("LOGTAIL_SOURCE_TOKEN")
LOGTAIL_URL = os.environ.get("LOGTAIL_URL")

REDIS_URL = os.environ.get("REDIS_URL")

LOCAL_LOG_DIR = Path(os.environ.get("LOCAL_LOG_DIR", "./logs"))
LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_LOG_FILE = LOCAL_LOG_DIR / "app_local.log"

# Redis list key
REDIS_LOG_KEY = "tavily_logs"

# ---------------------------------------------------------
# OPTIONAL: Redis wrapper
# ---------------------------------------------------------

def get_redis_client():
    """Returns a Redis/Valkey client if available, otherwise None."""
    try:
        if REDIS_URL:
            client = valkey.from_url(REDIS_URL, decode_responses=True)
            print(f"Trying to connect to Redis at {REDIS_URL}")
        else:
            client = valkey.Redis(host="localhost", port=6379, db=0, decode_responses=True)
            print("Trying to connect to local Redis at localhost:6379")

        client.ping() # if this fails, an exception will be raised
        print(f"Redis ping successful - {client}.")
        return client
    except Exception:
        print("Could not connect to Redis.")
        return None


# ---------------------------------------------------------
# CUSTOM REDIS HANDLER
# ---------------------------------------------------------

class RedisLogHandler(logging.Handler):
    """Pushes logs to a capped Redis list."""
    def __init__(self, client, key, max_entries=500):
        super().__init__()
        self.client = client
        self.key = key
        self.max_entries = max_entries

    def emit(self, record):
        """
        Takes a log record, formats it, and pushes it to Redis.
        """
        try:
            # Format the log record into a string
            message = self.format(record)
            # Push the entry to the left of the list
            self.client.lpush(self.key, message)
            # Trim the list to maintain the max size
            self.client.ltrim(self.key, 0, self.max_entries - 1)
        except Exception:
            pass   # do not break the app on logging error

# ---------------------------------------------------------
# LOGGER FACTORY
# ---------------------------------------------------------

def create_logger(name, handlers):
    """Create an isolated logger with given handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    for h in handlers:
        logger.addHandler(h)
    return logger


# ---------------------------------------------------------
# HANDLER SETUP
# ---------------------------------------------------------

# Local file handler
file_handler = logging.FileHandler(LOCAL_LOG_FILE, mode="a")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Logtail handler (BetterStack)
if LOGTAIL_TOKEN:
    logtail_handler = LogtailHandler(source_token=LOGTAIL_TOKEN, host=LOGTAIL_URL)
    logtail_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    print("Logtail handler configured.")
else:
    logtail_handler = None
    print("Logtail handler not configured.")

# Redis handler
redis_client = get_redis_client()
if redis_client:
    redis_handler = RedisLogHandler(redis_client, REDIS_LOG_KEY)
    redis_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    print("Redis handler configured.")
else:
    redis_handler = None
    print(f"Redis handler not configured")


# ---------------------------------------------------------
# PUBLIC LOGGERS
# ---------------------------------------------------------

logger_local = create_logger("local", [file_handler])

logger_betterstack = create_logger(
    "betterstack",
    [logtail_handler] if logtail_handler else []
)

logger_redis = create_logger(
    "redis",
    [redis_handler] if redis_handler else []
)

# MASTER LOGGER (optional)
logger_all = create_logger(
    "all",
    [h for h in [file_handler, logtail_handler, redis_handler] if h]
)

# Map used by get_logger()
_LOGGER_MAP = {
    "local": logger_local,
    "betterstack": logger_betterstack,
    "redis": logger_redis,
    "all": logger_all,
}


# ---------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------

def get_logger(name: str):
    """
    Returns one of the configured loggers:
      - 'local'
      - 'betterstack'
      - 'redis'
      - 'all'
    """
    return _LOGGER_MAP.get(name)
