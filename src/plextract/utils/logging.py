import logging
import os

# Read log level from environment, default to INFO
LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()

# Map string like "DEBUG" to logging.DEBUG, fallback to INFO on invalid values
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("plextract")