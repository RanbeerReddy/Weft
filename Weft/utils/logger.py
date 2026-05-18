import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# =========================
# LOG DIRECTORY
# =========================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    f"{datetime.now().strftime('%Y-%m-%d')}.log"
)

# =========================
# LOGGER CONFIGURATION
# =========================

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(filename)s:%(lineno)d | "
    "%(funcName)s() | "
    "%(message)s"
)

formatter = logging.Formatter(LOG_FORMAT)

# =========================
# FILE HANDLER
# =========================

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5 * 1024 * 1024,  # 5MB
    backupCount=5,
    encoding="utf-8"
)

file_handler.setFormatter(formatter)

# =========================
# CONSOLE HANDLER
# =========================

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# =========================
# MAIN LOGGER
# =========================

logger = logging.getLogger("weft")

logger.setLevel(logging.INFO)

# Prevent duplicate logs
logger.propagate = False

# Avoid duplicate handlers on reload
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)