import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler


def setup_logger(name: str = None):
    logger = logging.getLogger(name)
    if not logger.handlers:  # Prevent duplicate handlers in multi-import scenarios
        handlers = [logging.StreamHandler(sys.stdout), TimedRotatingFileHandler("app.log", when="d", interval=1, backupCount=7)]
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        configured_level = os.getenv("VERTEX_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO"
        logger.setLevel(getattr(logging, configured_level.strip().upper(), logging.INFO))
    return logger
