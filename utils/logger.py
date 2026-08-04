# ==========================================================
# LOGGING
# ==========================================================
import logging
import platform
from pathlib import Path
from datetime import datetime
from utils.config import APP_VERSION_TAG

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

platform_name = platform.system().lower()
log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"{log_time}_{platform_name}_{APP_VERSION_TAG}.log"

logger = logging.getLogger("ADaCITra")

if not logger.handlers:
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)